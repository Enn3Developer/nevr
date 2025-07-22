pub mod device;
pub mod pipeline;
pub mod shader;
pub mod surface;
pub mod swapchain;

use crate::engine::vulkan::surface::VulkanSurface;
use ash::prelude::VkResult;
use ash::vk::{ApplicationInfo, InstanceCreateInfo, PhysicalDevice, PhysicalDeviceProperties};
use ash::{Entry, Instance, vk};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr::null;
use std::str::FromStr;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;

pub struct Vulkan {
    entry: Entry,
    instance: Instance,
}

impl Vulkan {
    pub fn new(create_info: VulkanInstanceCreateInfo) -> VkResult<Self> {
        let entry = unsafe { Entry::load().unwrap() };

        let (instance_create_info, _extensions, _layers) = create_info.as_instance_create_info();

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

        Ok(Self { entry, instance })
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn create_surface(&self, window: &Window) -> VkResult<VulkanSurface> {
        VulkanSurface::new(&self.entry, &self.instance, window)
    }

    pub fn physical_devices(&self) -> VkResult<Vec<PhysicalDevice>> {
        unsafe { self.instance.enumerate_physical_devices() }
    }

    pub fn find_physical_device(&self, surface: &VulkanSurface) -> Option<(PhysicalDevice, u32)> {
        unsafe {
            self.physical_devices().ok()?.iter().find_map(|p| {
                self.instance
                    .get_physical_device_queue_family_properties(*p)
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        let supports_graphic_and_surface =
                            info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && surface
                                    .loader()
                                    .get_physical_device_surface_support(
                                        *p,
                                        index as u32,
                                        *surface.surface(),
                                    )
                                    .unwrap();
                        if supports_graphic_and_surface {
                            Some((*p, index as u32))
                        } else {
                            None
                        }
                    })
            })
        }
    }

    pub fn physical_device_properties(
        &self,
        physical_device: &PhysicalDevice,
    ) -> PhysicalDeviceProperties {
        unsafe {
            self.instance
                .get_physical_device_properties(*physical_device)
        }
    }

    pub fn physical_device_memory_properties(
        &self,
        physical_device: &PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(*physical_device)
        }
    }

    pub fn add_raytracing_properties(
        &self,
        physical_device: &PhysicalDevice,
        raytracing_properties: &mut vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    ) -> vk::PhysicalDeviceProperties2 {
        let properties = self.physical_device_properties(physical_device);

        let rt_properties: *mut vk::PhysicalDeviceRayTracingPipelinePropertiesKHR =
            raytracing_properties;

        let mut properties_2 = vk::PhysicalDeviceProperties2 {
            properties,
            p_next: rt_properties.cast::<c_void>(),
            ..Default::default()
        };

        unsafe {
            self.instance
                .get_physical_device_properties2(*physical_device, &mut properties_2);
        }

        properties_2
    }
}

pub struct VulkanVersion(pub(crate) u32);

impl From<(u32, u32, u32)> for VulkanVersion {
    fn from(value: (u32, u32, u32)) -> Self {
        Self(vk::make_api_version(0, value.0, value.1, value.2))
    }
}

impl From<u32> for VulkanVersion {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<(u32, u32, u32, u32)> for VulkanVersion {
    fn from(value: (u32, u32, u32, u32)) -> Self {
        Self(vk::make_api_version(value.0, value.1, value.2, value.3))
    }
}

impl From<&VulkanVersion> for VulkanVersion {
    fn from(value: &VulkanVersion) -> Self {
        Self(value.0)
    }
}

pub struct VulkanApplicationInfo {
    application_name: Option<CString>,
    engine_name: Option<CString>,
    vulkan_version: u32,
    engine_version: u32,
    application_version: u32,
}

impl VulkanApplicationInfo {
    pub fn with_application_name(mut self, application_name: impl AsRef<str>) -> Self {
        self.application_name = Some(CString::from_str(application_name.as_ref()).unwrap());

        self
    }

    pub fn with_engine_name(mut self, engine_name: impl AsRef<str>) -> Self {
        self.engine_name = Some(CString::from_str(engine_name.as_ref()).unwrap());

        self
    }

    pub fn with_vulkan_version(mut self, vulkan_version: impl Into<VulkanVersion>) -> Self {
        self.vulkan_version = vulkan_version.into().0;

        self
    }

    pub fn with_engine_version(mut self, engine_version: impl Into<VulkanVersion>) -> Self {
        self.engine_version = engine_version.into().0;

        self
    }

    pub fn with_app_version(mut self, app_version: impl Into<VulkanVersion>) -> Self {
        self.application_version = app_version.into().0;

        self
    }

    pub(crate) fn as_app_info(&self) -> ApplicationInfo {
        let p_engine_name = if let Some(engine_name) = &self.engine_name {
            engine_name.as_ptr()
        } else {
            null()
        };

        let p_application_name = if let Some(application_name) = &self.application_name {
            application_name.as_ptr()
        } else {
            null()
        };

        ApplicationInfo {
            application_version: self.application_version,
            engine_version: self.engine_version,
            api_version: self.vulkan_version,
            p_engine_name,
            p_application_name,
            ..Default::default()
        }
    }
}

impl Default for VulkanApplicationInfo {
    fn default() -> Self {
        Self {
            application_name: None,
            engine_name: None,
            vulkan_version: vk::make_api_version(0, 1, 3, 0),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            application_version: vk::make_api_version(0, 1, 0, 0),
        }
    }
}

pub struct VulkanInstanceCreateInfo {
    app_info: VulkanApplicationInfo,
    extensions: Vec<CString>,
    layers: Vec<CString>,
}

impl VulkanInstanceCreateInfo {
    pub fn with_app_info(mut self, app_info: VulkanApplicationInfo) -> Self {
        self.app_info = app_info;

        self
    }

    pub fn with_extensions<S: AsRef<str>>(
        mut self,
        extensions: impl IntoIterator<Item = S>,
    ) -> Self {
        self.extensions = extensions
            .into_iter()
            .map(|s| CString::from_str(s.as_ref()).unwrap())
            .collect();

        self
    }

    pub fn with_layers<S: AsRef<str>>(mut self, layers: impl IntoIterator<Item = S>) -> Self {
        self.layers = layers
            .into_iter()
            .map(|s| CString::from_str(s.as_ref()).unwrap())
            .collect();

        self
    }

    pub fn add_extension(mut self, extension: impl AsRef<str>) -> Self {
        self.extensions
            .push(CString::from_str(extension.as_ref()).unwrap());

        self
    }

    pub fn add_layer(mut self, layer: impl AsRef<str>) -> Self {
        self.layers.push(CString::from_str(layer.as_ref()).unwrap());

        self
    }

    pub fn enable_debug(self) -> Self {
        self.add_layer("VK_LAYER_KHRONOS_validation")
            .add_extension("VK_EXT_debug_utils")
    }

    pub fn add_required_extensions(mut self, window: &Window) -> Self {
        let mut extensions =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().into())
                .unwrap()
                .into_iter()
                .map(|e| unsafe { CStr::from_ptr(*e) })
                .map(|e| e.to_str().unwrap())
                .map(|e| CString::from_str(e).unwrap())
                .collect::<Vec<_>>();

        self.extensions.append(&mut extensions);

        self
    }

    pub(crate) fn as_instance_create_info(
        &self,
    ) -> (InstanceCreateInfo, Vec<*const c_char>, Vec<*const c_char>) {
        let extensions = self
            .extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();
        let layers = self.layers.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();

        (
            InstanceCreateInfo {
                p_application_info: &self.app_info.as_app_info(),
                enabled_extension_count: extensions.len() as u32,
                pp_enabled_extension_names: extensions.as_ptr(),
                enabled_layer_count: layers.len() as u32,
                pp_enabled_layer_names: layers.as_ptr(),
                ..Default::default()
            },
            extensions,
            layers,
        )
    }
}

impl Default for VulkanInstanceCreateInfo {
    fn default() -> Self {
        Self {
            app_info: VulkanApplicationInfo::default(),
            extensions: vec![],
            layers: vec![],
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
