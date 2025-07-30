#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use std::arch::aarch64::*;

    // 1. The CPU supports NEON.
    // 2. `a`, `b`, and `result` are all the same length.
    #[target_feature(enable = "neon")]
    pub unsafe fn add_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        
        while i + 4 <= len {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vr = vaddq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(i), vr);
            }
            i += 4;
        }
        
        while i < len {
            result[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn multiply_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        while i + 4 <= len {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vr = vmulq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(i), vr);
            }
            i += 4;
        }

        while i < len {
            result[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn sub_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) {

        let len = a.len();

        let mut i = 0;

        while i + 4 <= len {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vr = vsubq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(i), vr);
            }
            i += 4;
        }

        while i < len {
            result[i] = a[i] - b[i];
            i += 1;
        }
    }


    #[target_feature(enable = "neon")]
    pub unsafe fn divide_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) {

        let len = a.len();

        let mut i = 0;

        while i + 4 <= len {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vr = vdivq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(i), vr);
            }
            i += 4;
        }

        while i < len {
            result[i] = a[i] / b[i];
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32_neon() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
                let mut result = vec![0.0f32; 8];
                let expected = vec![9.0f32, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0];

                aarch64::add_f32_neon(&a, &b, &mut result);
                assert_eq!(result, expected);
            }
        }
    }

}
