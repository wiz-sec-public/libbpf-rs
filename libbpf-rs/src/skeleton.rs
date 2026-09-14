use core::ffi::c_void;
use std::alloc::alloc_zeroed;
use std::alloc::dealloc;
use std::alloc::Layout;
use std::ffi::CString;
use std::mem::size_of;
use std::mem::MaybeUninit;
use std::os::raw::c_char;
use std::os::raw::c_ulong;
use std::ptr;
use std::ptr::addr_of;
use std::ptr::NonNull;

use libbpf_sys::bpf_link;
use libbpf_sys::bpf_map;
use libbpf_sys::bpf_map_skeleton;
use libbpf_sys::bpf_object;
use libbpf_sys::bpf_object_skeleton;
use libbpf_sys::bpf_prog_skeleton;
use libbpf_sys::bpf_program;

use crate::error::IntoError as _;
use crate::util;
use crate::AsRawLibbpf;
use crate::Error;
use crate::Object;
use crate::ObjectBuilder;
use crate::OpenObject;
use crate::Result;

#[derive(Debug)]
struct MapSkelConfig {
    name: String,
    map: Box<*mut bpf_map>,
    mmaped: Option<Box<*mut c_void>>,
}

#[derive(Debug)]
struct ProgSkelConfig {
    name: String,
    prog: Box<*mut bpf_program>,
    link: Box<*mut bpf_link>,
}

#[derive(Debug)]
enum ObjectData<'dat> {
    Borrowed(&'dat [u8]),
    #[cfg(feature = "zstd")]
    Mapped(memmap2::MmapMut),
}

impl AsRef<[u8]> for ObjectData<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Borrowed(data) => data,
            #[cfg(feature = "zstd")]
            Self::Mapped(data) => data,
        }
    }
}

/// Discard only whole pages within the supplied read-only file mapping.
///
/// The caller must ensure that discarded bytes can be restored from the file.
#[cfg(all(feature = "zstd", target_os = "linux"))]
unsafe fn discard_file_backed_data(data: &[u8]) {
    // SAFETY: sysconf does not access any caller-provided memory.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return;
    }
    let page_size = page_size as usize;
    let skip = (data.as_ptr() as usize).wrapping_neg() % page_size;
    let len = data.len().saturating_sub(skip) / page_size * page_size;
    if len != 0 {
        // SAFETY: This range contains only complete pages within data. The
        // caller guarantees that refaulting them restores their contents.
        // Failure to discard pages must not prevent opening the BPF object.
        unsafe {
            libc::madvise(
                data.as_ptr().add(skip).cast_mut().cast(),
                len,
                libc::MADV_DONTNEED,
            );
        }
    }
}

#[expect(missing_docs)]
#[derive(Debug)]
pub struct ObjectSkeletonConfigBuilder<'dat> {
    data: ObjectData<'dat>,
    obj: Box<*mut bpf_object>,
    name: Option<String>,
    maps: Vec<MapSkelConfig>,
    progs: Vec<ProgSkelConfig>,
}

fn str_to_cstring_and_pool(s: &str, pool: &mut Vec<CString>) -> Result<*const c_char> {
    let cname = util::str_to_cstring(s)?;
    let p = cname.as_ptr();
    pool.push(cname);

    Ok(p)
}

impl<'dat> ObjectSkeletonConfigBuilder<'dat> {
    /// Construct a new instance
    ///
    /// `object_data` is the contents of the `.o` from clang
    ///
    /// `p` is a reference to the pointer where `libbpf_sys::bpf_object` should be
    /// stored/retrieved
    pub fn new(object_data: &'dat [u8]) -> Self {
        Self {
            data: ObjectData::Borrowed(object_data),
            obj: Box::new(ptr::null_mut()),
            name: None,
            maps: Vec::new(),
            progs: Vec::new(),
        }
    }

    /// Decompress a zstd object into an anonymous mapping owned by this builder.
    ///
    /// `uncompressed_size` bounds the allocation and must match the original
    /// ELF size. Corrupt frames and size mismatches return an error.
    /// Opening the skeleton unmaps this buffer, returning its resident pages
    /// directly to the OS instead of leaving them in the allocator's cache.
    #[cfg(feature = "zstd")]
    pub fn from_zstd(object_data: &[u8], uncompressed_size: usize) -> Result<Self> {
        if uncompressed_size == 0 {
            return Err(Error::with_invalid_data("BPF object must not be empty"));
        }
        let mut data = memmap2::MmapMut::map_anon(uncompressed_size)?;
        let size = zstd::bulk::decompress_to_buffer(object_data, &mut data)?;
        if size != uncompressed_size {
            return Err(Error::with_invalid_data(
                "Decompressed BPF object size mismatch",
            ));
        }
        let mut builder = Self::new(&[]);
        builder.data = ObjectData::Mapped(data);
        Ok(builder)
    }

    /// Decompress a zstd object embedded in a read-only file mapping.
    ///
    /// On Linux, also discard the compressed input's resident pages after
    /// decompression. Partial pages at the boundaries are left untouched.
    /// Opening the skeleton again faults the original bytes back in.
    ///
    /// # Safety
    ///
    /// `object_data` must be backed by an immutable file, such as a read-only
    /// section of the executable. It must not refer to heap or anonymous memory,
    /// or pages with private modifications that would be lost when discarded.
    #[cfg(feature = "zstd")]
    pub unsafe fn from_zstd_embedded(
        object_data: &'static [u8],
        uncompressed_size: usize,
    ) -> Result<Self> {
        let builder = Self::from_zstd(object_data, uncompressed_size)?;
        #[cfg(target_os = "linux")]
        // SAFETY: The caller guarantees that the input can be refaulted from its file.
        unsafe {
            discard_file_backed_data(object_data);
        }
        Ok(builder)
    }

    #[expect(missing_docs)]
    pub fn name<T: AsRef<str>>(&mut self, name: T) -> &mut Self {
        self.name = Some(name.as_ref().to_string());
        self
    }

    /// Adds a map to the config
    ///
    /// Set `mmaped` to `true` if the map is mmap'able to userspace
    pub fn map<T: AsRef<str>>(&mut self, name: T, mmaped: bool) -> &mut Self {
        let m = if mmaped {
            Some(Box::new(ptr::null_mut()))
        } else {
            None
        };

        self.maps.push(MapSkelConfig {
            name: name.as_ref().to_string(),
            map: Box::new(ptr::null_mut()),
            mmaped: m,
        });

        self
    }

    /// Adds a prog to the config
    pub fn prog<T: AsRef<str>>(&mut self, name: T) -> &mut Self {
        self.progs.push(ProgSkelConfig {
            name: name.as_ref().to_string(),
            prog: Box::new(ptr::null_mut()),
            link: Box::new(ptr::null_mut()),
        });

        self
    }

    fn build_maps(
        maps: &mut [MapSkelConfig],
        s: &mut bpf_object_skeleton,
        string_pool: &mut Vec<CString>,
    ) -> Option<Layout> {
        if maps.is_empty() {
            return None;
        }

        s.map_cnt = maps.len() as i32;
        s.map_skel_sz = size_of::<bpf_map_skeleton>() as i32;

        let layout = Layout::array::<bpf_map_skeleton>(maps.len())
            .expect("Failed to allocate memory for maps skeleton");

        unsafe {
            s.maps = alloc_zeroed(layout) as *mut bpf_map_skeleton;
            for (i, map) in maps.iter_mut().enumerate() {
                let current_map = s.maps.add(i);

                // Opt to panic on error here. We've already allocated memory and we'd rather not
                // leak. Extremely unlikely to have invalid unicode anyways.
                (*current_map).name = str_to_cstring_and_pool(&map.name, string_pool)
                    .expect("Invalid unicode in map name");
                (*current_map).map = &mut *map.map;
                (*current_map).mmaped = if let Some(ref mut mmaped) = map.mmaped {
                    &mut **mmaped
                } else {
                    ptr::null_mut()
                };
            }
        }

        Some(layout)
    }

    fn build_progs(
        progs: &mut [ProgSkelConfig],
        s: &mut bpf_object_skeleton,
        string_pool: &mut Vec<CString>,
    ) -> Option<Layout> {
        if progs.is_empty() {
            return None;
        }

        s.prog_cnt = progs.len() as i32;
        s.prog_skel_sz = size_of::<bpf_prog_skeleton>() as i32;

        let layout = Layout::array::<bpf_prog_skeleton>(progs.len())
            .expect("Failed to allocate memory for progs skeleton");

        unsafe {
            s.progs = alloc_zeroed(layout) as *mut bpf_prog_skeleton;
            for (i, prog) in progs.iter_mut().enumerate() {
                let current_prog = s.progs.add(i);

                // See above for `expect()` rationale
                (*current_prog).name = str_to_cstring_and_pool(&prog.name, string_pool)
                    .expect("Invalid unicode in prog name");
                (*current_prog).prog = &mut *prog.prog;
                (*current_prog).link = &mut *prog.link;
            }
        }

        Some(layout)
    }

    #[expect(missing_docs)]
    pub fn build(mut self) -> Result<ObjectSkeletonConfig<'dat>> {
        // Holds `CString`s alive so pointers to them stay valid
        let mut string_pool = Vec::new();

        let mut s = libbpf_sys::bpf_object_skeleton {
            sz: size_of::<bpf_object_skeleton>() as c_ulong,
            ..Default::default()
        };

        if let Some(ref n) = self.name {
            s.name = str_to_cstring_and_pool(n, &mut string_pool)?;
        }

        // libbpf_sys will use it as const despite the signature
        s.data = self.data.as_ref().as_ptr() as *mut c_void;
        s.data_sz = self.data.as_ref().len() as c_ulong;

        // Give s ownership over the box
        s.obj = Box::into_raw(self.obj);

        let maps_layout = Self::build_maps(&mut self.maps, &mut s, &mut string_pool);
        let progs_layout = Self::build_progs(&mut self.progs, &mut s, &mut string_pool);

        Ok(ObjectSkeletonConfig {
            inner: s,
            maps: self.maps,
            progs: self.progs,
            maps_layout,
            progs_layout,
            _data: self.data,
            _string_pool: string_pool,
        })
    }
}

/// Helper struct that wraps a `libbpf_sys::bpf_object_skeleton`.
///
/// This struct will:
/// * ensure lifetimes are valid for dependencies (pointers, data buffer)
/// * free any allocated memory on drop
///
/// This struct can be moved around at will. Upon drop, all allocated resources will be freed
/// except the BPF object, which belongs to the generated skeleton's `OpenObject` or `Object`.
/// [`Self::open`] releases the ELF buffer after libbpf has read its contents.
#[derive(Debug)]
pub struct ObjectSkeletonConfig<'dat> {
    inner: bpf_object_skeleton,
    maps: Vec<MapSkelConfig>,
    progs: Vec<ProgSkelConfig>,
    /// Layout necessary to `dealloc` memory
    maps_layout: Option<Layout>,
    /// Same as above
    progs_layout: Option<Layout>,
    /// Keep the ELF bytes alive until libbpf has finished opening the object.
    _data: ObjectData<'dat>,
    /// Hold strings alive so pointers to them stay valid
    _string_pool: Vec<CString>,
}

impl ObjectSkeletonConfig<'_> {
    /// Open the skeleton and release its ELF buffer on success.
    ///
    /// The opened BPF object must be taken into an `OpenObject` by the caller.
    /// This config does not own that object. A second open returns an error.
    ///
    /// # Safety
    ///
    /// `opts` must be null or point to valid libbpf open options, including any
    /// pointers stored within the options, for the duration of this call.
    pub unsafe fn open(&mut self, opts: *const libbpf_sys::bpf_object_open_opts) -> Result<()> {
        // SAFETY: `build` allocates the object pointer and keeps it alive.
        if !unsafe { *self.inner.obj }.is_null() {
            return Err(Error::with_invalid_data("Skeleton is already open"));
        }
        // SAFETY: All skeleton storage is owned by self; opts is valid by contract.
        let ret = unsafe { libbpf_sys::bpf_object__open_skeleton(&mut self.inner, opts) };
        if ret != 0 {
            // libbpf can fail to populate maps or programs after opening the
            // object. No OpenObject exists yet to close it in that case.
            unsafe {
                libbpf_sys::bpf_object__close(*self.inner.obj);
                *self.inner.obj = ptr::null_mut();
            }
            return Err(Error::from_raw_os_error(-ret));
        }

        // libbpf finishes ELF parsing in bpf_object_open; load and attach only
        // use the resulting object, maps and programs, not these input bytes.
        self.inner.data = ptr::null_mut();
        self.inner.data_sz = 0;
        self._data = ObjectData::Borrowed(&[]);
        Ok(())
    }

    /// Returns the `mmaped` pointer for a map at the specified `index`.
    ///
    /// The index is determined by the order in which the map was passed to
    /// `ObjectSkeletonConfigBuilder::map`. Index starts at 0.
    ///
    /// Warning: the returned pointer is only valid while the `ObjectSkeletonConfig` is alive.
    pub fn map_mmap_ptr(&self, index: usize) -> Result<*mut c_void> {
        if index >= self.maps.len() {
            return Err(Error::with_invalid_data(format!(
                "Invalid map index: {index}"
            )));
        }

        let p = self.maps[index]
            .mmaped
            .as_ref()
            .ok_or_invalid_data(|| "Map does not have mmaped ptr")?;
        Ok(**p)
    }

    /// Returns the link pointer for a prog at the specified `index`.
    ///
    /// The index is determined by the order in which the prog was passed to
    /// `ObjectSkeletonConfigBuilder::prog`. Index starts at 0.
    ///
    /// Warning: the returned pointer is only valid while the `ObjectSkeletonConfig` is alive.
    pub fn prog_link_ptr(&self, index: usize) -> Result<*mut bpf_link> {
        if index >= self.progs.len() {
            return Err(Error::with_invalid_data(format!(
                "Invalid prog index: {index}"
            )));
        }

        Ok(*self.progs[index].link)
    }
}

impl AsRawLibbpf for ObjectSkeletonConfig<'_> {
    type LibbpfType = libbpf_sys::bpf_object_skeleton;

    /// Retrieve the underlying [`libbpf_sys::bpf_object_skeleton`].
    fn as_libbpf_object(&self) -> NonNull<Self::LibbpfType> {
        // SAFETY: A reference is always a valid pointer.
        unsafe { NonNull::new_unchecked(addr_of!(self.inner).cast_mut()) }
    }
}

impl Drop for ObjectSkeletonConfig<'_> {
    // Note we do *not* run `libbpf_sys::bpf_object__destroy_skeleton` here.
    //
    // Couple reasons:
    //
    // 1) We did not allocate `libbpf_sys::bpf_object_skeleton` on the heap and
    //    `libbpf_sys::bpf_object__destroy_skeleton` will try to free from heap
    //
    // 2) `libbpf_object_skeleton` assumes it "owns" the object and everything inside it.
    //    libbpf-cargo's generated skeleton instead gives ownership of the object to
    //    libbpf-rs::*Object. The destructors in libbpf-rs::*Object will know when and how to do
    //    cleanup.
    fn drop(&mut self) {
        assert_eq!(self.maps_layout.is_none(), self.inner.maps.is_null());
        assert_eq!(self.progs_layout.is_none(), self.inner.progs.is_null());

        if let Some(layout) = self.maps_layout {
            unsafe {
                dealloc(self.inner.maps as _, layout);
            }
        }

        if let Some(layout) = self.progs_layout {
            unsafe {
                dealloc(self.inner.progs as _, layout);
            }
        }

        let () = drop(unsafe { Box::from_raw(self.inner.obj) });
    }
}

/// A trait for skeleton builder.
pub trait SkelBuilder<'obj> {
    /// Define that when BPF object is opened, the returned type should implement the [`OpenSkel`]
    /// trait
    type Output: OpenSkel<'obj>;

    /// Open eBPF object and return [`OpenSkel`]
    fn open(self, object: &'obj mut MaybeUninit<OpenObject>) -> Result<Self::Output>;

    /// Open eBPF object with [`libbpf_sys::bpf_object_open_opts`] and return [`OpenSkel`]
    fn open_opts(
        self,
        open_opts: libbpf_sys::bpf_object_open_opts,
        object: &'obj mut MaybeUninit<OpenObject>,
    ) -> Result<Self::Output>;

    /// Get a reference to [`ObjectBuilder`]
    fn object_builder(&self) -> &ObjectBuilder;

    /// Get a mutable reference to [`ObjectBuilder`]
    fn object_builder_mut(&mut self) -> &mut ObjectBuilder;
}

/// A trait for opened skeleton.
///
/// In addition to the methods defined in this trait, skeletons that implement this trait will also
/// have bespoke implementations of a few additional methods to facilitate access to global
/// variables of the BPF program. These methods will be named `bss()`, `data()`, and `rodata()`.
/// Each corresponds to the variables stored in the BPF ELF program section of the same name.
/// However if your BPF program lacks one of these sections the corresponding rust method will not
/// be generated.
///
/// The type of the value returned by each of these methods will be specific to your BPF program.
/// A common convention is to define a single global variable in the BPF program with a struct type
/// containing a field for each configuration parameter <sup>\[[source]\]</sup>. libbpf-rs
/// auto-generates this pattern for you without you having to define such a struct type in your BPF
/// program. It does this by examining each of the global variables in your BPF program's `.bss`,
/// `.data`, and `.rodata` sections and then creating Rust struct types. Since these struct types
/// are specific to the layout of your BPF program, they are not documented in this crate. However
/// you can see documentation for them by running `cargo doc` in your own project and looking at
/// the `imp` module. You can also view their implementation by looking at the generated skeleton
/// rust source file. The use of these methods can also be seen in the examples 'capable',
/// 'runqslower', and 'tproxy'.
///
/// If you ever doubt whether libbpf-rs has placed a particular variable in the correct struct
/// type, you can see which section each global variable is stored in by examining the output of
/// the following command (after a successful build):
///
/// ```sh
/// bpf-objdump --syms ./target/bpf/*.bpf.o
/// ```
///
/// [source]: https://nakryiko.com/posts/bcc-to-libbpf-howto-guide/#application-configuration
pub trait OpenSkel<'obj> {
    /// Define that when BPF object is loaded, the returned type should implement the [`Skel`] trait
    type Output: Skel<'obj>;

    /// Load BPF object and return [`Skel`].
    fn load(self) -> Result<Self::Output>;

    /// Get a reference to [`OpenObject`].
    fn open_object(&self) -> &OpenObject;

    /// Get a mutable reference to [`OpenObject`].
    fn open_object_mut(&mut self) -> &mut OpenObject;
}

/// A trait for loaded skeleton.
pub trait Skel<'obj> {
    /// Attach BPF object.
    fn attach(&mut self) -> Result<()> {
        unimplemented!()
    }
    /// Get a reference to [`Object`].
    fn object(&self) -> &Object;

    /// Get a mutable reference to [`Object`].
    fn object_mut(&mut self) -> &mut Object;
}

#[cfg(all(test, feature = "zstd"))]
mod tests {
    use super::ObjectSkeletonConfigBuilder;
    use std::ptr;

    #[test]
    fn compressed_object_rejects_corruption_and_size_mismatches() {
        let data = b"BPF object data";
        let compressed = zstd::bulk::compress(data, 19).unwrap();
        let builder = ObjectSkeletonConfigBuilder::from_zstd(&compressed, data.len()).unwrap();
        assert_eq!(builder.data.as_ref(), data);
        assert!(ObjectSkeletonConfigBuilder::from_zstd(b"invalid", data.len()).is_err());
        assert!(ObjectSkeletonConfigBuilder::from_zstd(
            &compressed[..compressed.len() / 2],
            data.len(),
        )
        .is_err());
        assert!(ObjectSkeletonConfigBuilder::from_zstd(&compressed, data.len() - 1).is_err());
        assert!(ObjectSkeletonConfigBuilder::from_zstd(&compressed, data.len() + 1).is_err());
        assert!(ObjectSkeletonConfigBuilder::from_zstd(&compressed, 0).is_err());
    }

    #[test]
    fn compressed_object_rejects_invalid_elf() {
        let data = b"not an ELF";
        let compressed = zstd::bulk::compress(data, 19).unwrap();
        let mut builder = ObjectSkeletonConfigBuilder::from_zstd(&compressed, data.len()).unwrap();
        builder.name("invalid_bpf");
        let mut config = builder.build().unwrap();
        // SAFETY: Null opts selects libbpf defaults.
        assert!(unsafe { config.open(ptr::null()) }.is_err());
        // SAFETY: The config owns the object pointer storage.
        assert!(unsafe { *config.inner.obj }.is_null());
    }

    #[cfg(target_os = "linux")]
    #[test_fork::fork]
    #[test]
    fn decompressed_object_is_unmapped_on_drop() {
        // SAFETY: sysconf does not access caller-provided memory.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let data = vec![42; page_size * 2];
        let compressed = zstd::bulk::compress(&data, 19).unwrap();
        let builder = ObjectSkeletonConfigBuilder::from_zstd(&compressed, data.len()).unwrap();
        let addr = builder.data.as_ref().as_ptr().cast_mut().cast();
        let mut residency = [0u8; 2];
        // SAFETY: The mapping is page aligned, and residency has one byte per page.
        assert_eq!(
            unsafe { libc::mincore(addr, data.len(), residency.as_mut_ptr()) },
            0
        );
        assert_eq!(residency, [1, 1]);
        drop(builder);
        // SAFETY: mincore queries virtual addresses without dereferencing them.
        assert_eq!(
            unsafe { libc::mincore(addr, data.len(), residency.as_mut_ptr()) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ENOMEM)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn discard_preserves_partial_pages_and_refaults_file_contents() {
        use std::io::Write;

        // SAFETY: sysconf does not access caller-provided memory.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let data = vec![42; page_size * 5];
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&data).unwrap();
        // SAFETY: This test owns the file, and it is not modified while mapped.
        let mut mapping = unsafe { memmap2::MmapOptions::new().map_copy(&file) }.unwrap();
        mapping[0] = 1;
        mapping[data.len() - 1] = 2;
        assert_eq!(&mapping[1..data.len() - 1], &data[1..data.len() - 1]);
        // SAFETY: Only the complete interior pages are discarded, and they
        // contain unmodified file data. The modified boundary pages must survive.
        unsafe { super::discard_file_backed_data(&mapping[1..data.len() - 1]) };
        assert_eq!(mapping[0], 1);
        assert_eq!(mapping[data.len() - 1], 2);
        assert_eq!(&mapping[1..data.len() - 1], &data[1..data.len() - 1]);
    }
}
