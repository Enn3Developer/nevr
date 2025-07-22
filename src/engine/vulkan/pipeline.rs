use crate::vulkan::shader::VulkanShader;
use ash::prelude::VkResult;
use ash::vk::{
    DeferredOperationKHR, DescriptorSetLayout, Pipeline, PipelineCache, PipelineLayout,
    RayTracingShaderGroupTypeKHR, ShaderStageFlags,
};
use ash::{Device, Instance, vk};
use std::mem::MaybeUninit;
use std::ptr::null;

pub struct VulkanPipeline {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
}

impl VulkanPipeline {
    pub fn new(
        instance: &Instance,
        descriptor_set_layouts: &[DescriptorSetLayout],
        device: &Device,
        shaders: [&VulkanShader; 4],
    ) -> VkResult<Self> {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        let main_fn = c"main";

        let pipeline_shader_stage_create_infos = vec![
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::CLOSEST_HIT_KHR,
                module: shaders[1].shader_module,
                p_name: main_fn.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::RAYGEN_KHR,
                module: shaders[0].shader_module,
                p_name: main_fn.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::MISS_KHR,
                module: shaders[2].shader_module,
                p_name: main_fn.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::MISS_KHR,
                module: shaders[3].shader_module,
                p_name: main_fn.as_ptr(),
                ..Default::default()
            },
        ];

        let raytracing_shader_group_create_infos = vec![
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty: RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
                general_shader: vk::SHADER_UNUSED_KHR,
                closest_hit_shader: 0,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty: RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader: 1,
                closest_hit_shader: vk::SHADER_UNUSED_KHR,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty: RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader: 2,
                closest_hit_shader: vk::SHADER_UNUSED_KHR,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty: RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader: 3,
                closest_hit_shader: vk::SHADER_UNUSED_KHR,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
        ];

        let raytracing_pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR {
            stage_count: 4,
            p_stages: pipeline_shader_stage_create_infos.as_ptr(),
            group_count: 4,
            p_groups: raytracing_shader_group_create_infos.as_ptr(),
            max_pipeline_ray_recursion_depth: 1,
            layout: pipeline_layout,
            ..Default::default()
        };

        let function_name = c"vkCreateRayTracingPipelinesKHR";

        let pvk_create_raytracing_pipeline_khr: vk::PFN_vkCreateRayTracingPipelinesKHR = unsafe {
            instance
                .get_device_proc_addr(device.handle(), function_name.as_ptr())
                .map(|func| std::mem::transmute(func))
        }
        .unwrap();

        let mut pipeline = MaybeUninit::uninit();

        let result = unsafe {
            pvk_create_raytracing_pipeline_khr(
                device.handle(),
                DeferredOperationKHR::null(),
                PipelineCache::null(),
                1,
                &raytracing_pipeline_create_info,
                null(),
                pipeline.as_mut_ptr(),
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(result);
        }

        let pipeline = unsafe { pipeline.assume_init() };

        Ok(Self {
            pipeline,
            pipeline_layout,
            device: device.clone(),
        })
    }
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
