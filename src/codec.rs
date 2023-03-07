use tfhe::core_crypto::prelude::*;

#[inline]
pub const fn log2(input: usize) -> usize {
    core::mem::size_of::<usize>() * 8 - (input.leading_zeros() as usize) - 1
}

pub struct Codec {
    decomposer: SignedDecomposer<u64>,
    delta: u64,
}

impl Codec {
    pub fn new(t: u64) -> Self {
        if !t.is_power_of_two() {
            panic!("delta must be a power of 2")
        }
        let logt = log2(t as usize);
        let decomposer =
            SignedDecomposer::<u64>::new(DecompositionBaseLog(logt), DecompositionLevelCount(1));
        Self {
            decomposer,
            delta: 1 << (u64::BITS as usize - logt),
        }
    }

    pub const fn largest_error(&self) -> u64 {
        self.delta / 2
    }

    pub fn pt_modulus_bits(&self) -> usize {
        self.decomposer.base_log().0
    }

    pub fn pt_modulus(&self) -> u64 {
        1 << self.decomposer.base_log().0
    }

    pub fn encode(&self, x: &mut u64) {
        if *x >= self.pt_modulus() {
            panic!("value is too big")
        }
        *x *= self.delta
    }

    pub fn decode(&self, x: &mut u64) {
        let tmp = self.decomposer.closest_representable(*x);
        *x = tmp / self.delta
    }

    /// Encode ternary x as x*(q/3)
    pub fn ternary_encode(x: &mut u64) {
        const THIRD: u64 = (u64::MAX as f64 / 3.0) as u64;
        if *x == 0 {
            *x = 0;
        } else if *x == 1 {
            *x = THIRD;
        } else if *x == u64::MAX {
            *x = 2 * THIRD;
        } else {
            panic!("not a ternary scalar")
        }
    }

    pub fn ternary_decode(x: &mut u64) {
        const SIXTH: u64 = (u64::MAX as f64 / 6.0) as u64;
        const THIRD: u64 = SIXTH + SIXTH;
        const HALF: u64 = u64::MAX / 2;
        if *x > SIXTH && *x <= HALF {
            *x = 1;
        } else if *x > HALF && *x <= HALF + THIRD {
            *x = u64::MAX;
        } else {
            *x = 0;
        }
    }

    /*
    /// Encode a polynomial.
    pub fn poly_encode<C>(&self, xs: &mut Polynomial<C>)
        where
            C: AsMutSlice<Element = u64>,
    {
        for coeff in xs.coefficient_iter_mut() {
            self.encode(coeff);
        }
    }

    pub fn poly_decode<C>(&self, xs: &mut Polynomial<C>)
        where
            C: AsMutSlice<Element = u64>,
    {
        for coeff in xs.coefficient_iter_mut() {
            self.decode(coeff);
        }
    }

    /// Encode a ternary polynomial.
    pub fn poly_ternary_encode<C>(xs: &mut Polynomial<C>)
        where
            C: AsMutSlice<Element = u64>,
    {
        for coeff in xs.coefficient_iter_mut() {
            Self::ternary_encode(coeff);
        }
    }

    pub fn poly_ternary_decode<C>(xs: &mut Polynomial<C>)
        where
            C: AsMutSlice<Element = u64>,
    {
        for coeff in xs.coefficient_iter_mut() {
            Self::ternary_decode(coeff);
        }
    }
     */
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_encoder() {
        let codec = Codec::new(1 << 1);
        {
            let mut x: u64 = 0;
            codec.encode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: u64 = 1;
            codec.encode(&mut x);
            assert_eq!(x, 1 << (u64::BITS - 1));
        }
        {
            let mut x: u64 = 10;
            codec.decode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: u64 = u64::MAX;
            codec.decode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: u64 = 1 << (u64::BITS - 1);
            codec.decode(&mut x);
            assert_eq!(x, 1);
        }
    }

    #[test]
    fn test_generic_encoder() {
        {
            let expected = 3;
            let codec = Codec::new(4);
            let mut encoded = expected;
            codec.encode(&mut encoded);

            let mut decoded1 = encoded + codec.largest_error() - 1;
            let mut decoded2 = encoded - codec.largest_error() + 1;

            codec.decode(&mut decoded1);
            codec.decode(&mut decoded2);

            assert_eq!(decoded1, expected);
            assert_eq!(decoded2, expected);
        }

        {
            let expected = 1 << 6;
            let codec = Codec::new(1 << 12);
            let mut encoded = expected;
            codec.encode(&mut encoded);

            let mut decoded1 = encoded + codec.largest_error() - 1;
            let mut decoded2 = encoded - codec.largest_error() + 1;

            codec.decode(&mut decoded1);
            codec.decode(&mut decoded2);

            assert_eq!(decoded1, expected);
            assert_eq!(decoded2, expected);
        }
    }
}
