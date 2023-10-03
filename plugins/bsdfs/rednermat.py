import numpy as np
import mitsuba as mi
import drjit as dr
import math
from typing import Tuple

# Redner material implementation inside Mitsuba
# We are assuming all component used
class RednerMat(mi.BSDF):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        
        self.albedo = props.get("albedo", mi.Texture.D65(0.1))
        self.roughness = props.get("roughness", mi.Texture.D65(0.1))
        self.specular = props.get("specular", mi.Texture.D65(0.1))
        
        self.m_components += [mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide]
        self.m_components += [mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide]
        self.m_flags = self.m_components[0] | self.m_components[1]
        
    def roughness_to_phong(roughness: mi.Float) -> mi.Float:
        return dr.maximum(2.0 / roughness - 2, 0)
    
    def smithG1(roughness: mi.Float, v: mi.Vector3f) -> mi.Float:
        cos_theta = v.z
        # tan^2 + 1 = 1/cos^2
        tan_theta = dr.safe_sqrt(dr.rcp(cos_theta * cos_theta) - 1.0)
        
        a =  dr.rcp(dr.sqrt(roughness) * tan_theta)
        result = dr.select(
            a > 1.6, 1, 
            (3.535 * a + 2.181 * a * a) / (1.0 + 2.276 * a + 2.577 * a*a)
        )
        return dr.select(dr.eq(tan_theta,0.0), 1, result)
    
    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True) -> mi.Color3f:
        has_diffuse = ctx.is_enabled(mi.BSDFFlags.DiffuseReflection, 0)
        has_specular = ctx.is_enabled(mi.BSDFFlags.GlossyReflection, 1)
        
        # Calcul contribution diffuse
        diffuse_contrib = dr.select(has_diffuse, self.albedo.eval(si, active) * dr.maximum(wo.z, 0.0) / mi.Float(math.pi), 0.0)
        
        # Calcul contribution speculaire        
        m = dr.normalize(si.wi + wo)
        roughness = dr.maximum(self.roughness.eval_1(si, active),0.00001)
        phong_exponent = RednerMat.roughness_to_phong(roughness)
        specular_reflectance = self.specular.eval(si, active)
        
        D = dr.power(dr.maximum(m.z, 0.0), phong_exponent) * (phong_exponent + 2.0) / mi.Float(2.0 * math.pi)
        G = RednerMat.smithG1(roughness, si.wi) * RednerMat.smithG1(roughness, wo)
        F = specular_reflectance + (1 - specular_reflectance) * dr.power(dr.maximum(1 - dr.abs(dr.dot(m, wo)), 0), 5)
        specular_contrib = dr.select(wo.z > 0, F * D * G / (4.0 * si.wi.z), 0)
        specular_contrib = dr.select(has_specular, specular_contrib, 0)
        
        return specular_contrib + diffuse_contrib
        
    
    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True) -> float:
        has_diffuse = ctx.is_enabled(mi.BSDFFlags.DiffuseReflection, 0)
        has_specular = ctx.is_enabled(mi.BSDFFlags.GlossyReflection, 1)
        
        # Compute the probability of selecting the diffuse and specular lobe
        diffuse_pmf = mi.luminance(self.albedo.eval(si, active))
        specular_pmf = mi.luminance(self.specular.eval(si, active))
        weight_pmf = diffuse_pmf + specular_pmf
        diffuse_pmf = dr.select(weight_pmf > 0.0, diffuse_pmf / weight_pmf, 0.0)
        specular_pmf = dr.select(weight_pmf > 0.0, specular_pmf / weight_pmf, 0.0)
        
        if(dr.neq(has_diffuse, has_specular)):
            diffuse_pmf = dr.select(has_diffuse, 1.0, 0.0)
            specular_pmf = dr.select(has_diffuse, 0.0, 1.0)
        
        # Compute diffuse PDF
        diffuse_pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo) * diffuse_pmf
        
        # Compute specular PDF
        m = dr.normalize(si.wi + wo)
        roughness = dr.maximum(self.roughness.eval_1(si, active),0.00001)
        phong_exponent = RednerMat.roughness_to_phong(roughness)
        D = dr.power(dr.maximum(m.z, 0.0), phong_exponent) * (phong_exponent + 2.0) / mi.Float(2 * math.pi)
        specular_pdf = dr.select(dr.abs(dr.dot(m, wo)) > 0,  specular_pmf * D * dr.maximum(m.z, 0.0) / (4.0 * dr.abs(dr.dot(m, wo))), 0.0)
        
        return dr.select(wo.z > 0, diffuse_pdf + specular_pdf, 0)
    
    def eval_pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True) -> Tuple[mi.Color3f, mi.Float]:
        # TODO: Could be more efficient by fusing some computations
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)
    
    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, sample1: mi.Float, sample2: mi.Point2f, active: bool = True) -> Tuple[mi.BSDFSample3f, mi.Color3f]:
        has_diffuse = ctx.is_enabled(mi.BSDFFlags.DiffuseReflection, 0)
        has_specular = ctx.is_enabled(mi.BSDFFlags.GlossyReflection, 1)
        
        # Compute PDF selection
        diffuse_pmf = mi.luminance(self.albedo.eval(si, active))
        specular_pmf = mi.luminance(self.specular.eval(si, active))
        weight_pmf = diffuse_pmf + specular_pmf
        diffuse_pmf = dr.select(weight_pmf > 0.0, diffuse_pmf / weight_pmf, 0.0)
        specular_pmf = dr.select(weight_pmf > 0.0, specular_pmf / weight_pmf, 0.0)
        
        if(dr.neq(has_diffuse, has_specular)):
            diffuse_pmf = dr.select(has_diffuse, 1.0, 0.0)
            specular_pmf = dr.select(has_diffuse, 0.0, 1.0)
             
        
        # Compute masks
        diffuse_mask = diffuse_pmf < sample1
        specular_mask = ~diffuse_mask
        
        # Example to do specular and diffuse sampling
        # https://github.com/mitsuba-renderer/mitsuba3/blob/6105dfc1057a3bc204393b8e9557bcb26e08568b/src/bsdfs/pplastic.cpp#L243C9-L256C10
        # Uses a lot of masks
        bs: mi.BSDFSample3f = dr.zeros(mi.BSDFSample3f)
        bs.eta = 1
        
        
        #if dr.any(diffuse_mask):
        bs.wo[diffuse_mask] = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.sampled_type[diffuse_mask]  = +mi.BSDFFlags.DiffuseReflection
        bs.sampled_component[diffuse_mask]  = 0
        
        roughness = dr.maximum(self.roughness.eval_1(si, active),0.00001)
        phong_exponent = RednerMat.roughness_to_phong(roughness)
        phi = 2.0 * math.pi * sample2[1]
        sin_phi = dr.sin(phi)
        cos_phi = dr.cos(phi)
        cos_theta = dr.power(sample2[0], dr.rcp(phong_exponent + 2.0))
        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)
        m = mi.Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
        bs.wo[specular_mask] = mi.reflect(si.wi, m)
        bs.sampled_type[specular_mask]  = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component[specular_mask]  = 1
        
        # Compute PDF
        bs.pdf = self.pdf(ctx, si, bs.wo, active)
        active &= bs.pdf > 0.0
        result = self.eval(ctx, si, bs.wo, active)
        
        return (bs, result / bs.pdf & active)
    
    def to_string(self):
        return ('RednerMat[\n'
                '    albedo=%s,\n'
                '    roughness=%s,\n'
                '    specular=%s,\n'
                ']' % (self.albedo, self.roughness, self.specular))
    
    def traverse(self, callback):
        callback.put_object('albedo', self.albedo, mi.ParamFlags.Differentiable)
        callback.put_object('roughness', self.roughness, mi.ParamFlags.Differentiable )
        callback.put_object('specular', self.specular, mi.ParamFlags.Differentiable)