use std::cmp;

pub struct BatcherSort<T>
where
    T: Ord,
{
    pub vs: Vec<T>,
    k: usize,
    comp_count: usize,
    verbose: bool,
}

impl<T> BatcherSort<T>
where
    T: Ord,
{
    pub fn new(vs: Vec<T>) -> Self {
        let k = vs.len();
        Self {
            vs,
            k,
            comp_count: 0,
            verbose: false,
        }
    }

    pub fn new_k(vs: Vec<T>, k: usize) -> Self {
        Self {
            vs,
            k,
            comp_count: 0,
            verbose: false,
        }
    }

    pub fn sort(&mut self) {
        self.sort_rec(0, self.vs.len());
    }

    fn sort_rec(&mut self, start: usize, len: usize) {
        if self.verbose {
            println!("[sort_rec begin] lo={}, n={}", start, len);
        }
        if len > 1 {
            let n = len / 2;
            let m = len - n;
            self.sort_rec(start, n);
            self.sort_rec(start + n, m);

            let indices: Vec<_> = (start..start + len).collect();
            let (ix_full, jx_full) = indices.split_at(n);

            let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
            let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
            self.merge_rec(&ix, &jx);
        }
        if self.verbose {
            println!("[sort_rec exit] lo={}, n={}", start, len);
        }
    }

    /// We assume the two sorted arrays we wish to merge are consecutive of length n+m,
    /// the first one has length `n` that's always even and the second one has `m`.
    pub fn merge(&mut self) {
        let n = self.vs.len() / 2;
        let m = self.vs.len() - n;
        // n = cmp::min(n, self.k);
        // m = cmp::min(m, self.k);

        let ix_full: Vec<_> = (0..n).collect();
        let jx_full: Vec<_> = (n..n + m).collect();
        let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
        let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
        self.merge_rec(&ix, &jx)
    }

    fn merge_rec(&mut self, ix: &[usize], jx: &[usize]) {
        if self.verbose {
            println!("[merge begin] ix={:?}, jx={:?}", ix, jx);
        }
        let nm = ix.len() * jx.len();
        if nm > 1 {
            let even_ix = self.even_indices(ix);
            let even_jx = self.even_indices(jx);
            let odd_ix = self.odd_indices(ix);
            let odd_jx = self.odd_indices(jx);
            self.merge_rec(&even_ix, &even_jx);
            self.merge_rec(&odd_ix, &odd_jx);

            let even_all = [even_ix, even_jx].concat();
            let odd_all = [odd_ix, odd_jx].concat();
            let tmp = ((even_all.len() as f64 / 2f64).floor()
                + (odd_all.len() as f64 / 2f64).floor()) as usize;
            let w_max = if even_all.len() % 2 == 0 && odd_all.len() % 2 == 0 {
                tmp - 1
            } else {
                tmp
            };
            for i in 0..w_max {
                // NOTE maybe we can break early
                if odd_all[i] < self.k || even_all[i + 1] < self.k {
                    self.compare_at(odd_all[i], even_all[i + 1]);
                }
            }

            // the final output is v1, w1, v2, w2...
            // correction needed if `|ix|` is odd
            if ix.len() % 2 == 1 {
                let end = if jx.len() % 2 == 0 {
                    jx.len()
                } else {
                    jx.len() - 1
                };
                for i in (0..end).step_by(2) {
                    self.vs.swap(jx[i], jx[i + 1]);
                }
            }
        } else if nm == 1 {
            self.compare_at(ix[0], jx[0]);
        } else {
            // do nothing because we have 1 or 0 elements
        }
        if self.verbose {
            println!("[merge exit] ix={:?}, jx={:?}", ix, jx);
        }
    }

    /// Swap in-place an elements at index `i` with another at index `j`
    fn compare_at(&mut self, i: usize, j: usize) {
        if self.verbose {
            println!("[compare_at] i={}, j={}", i, j);
        }
        self.comp_count += 1;
        if self.vs[i] > self.vs[j] {
            self.vs.swap(i, j);
        }
    }

    /// Output the number of comparisons
    pub fn comparisons(&self) -> usize {
        self.comp_count
    }

    fn odd_indices(&self, indices: &[usize]) -> Vec<usize> {
        let new_indices = indices.split_first().unwrap().1;
        self.even_indices(new_indices)
    }

    fn even_indices(&self, indices: &[usize]) -> Vec<usize> {
        let mut out = vec![];
        for i in (0..indices.len()).step_by(2) {
            if i < self.k {
                out.push(indices[i]);
            }
        }
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[test]
    fn test_merge_2() {
        {
            let mut batcher = BatcherSort::new(vec![2, 1]);
            batcher.merge();
            assert_eq!(vec![1, 2], batcher.vs);
            assert_eq!(1, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(vec![2, 1], k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_3() {
        {
            let mut batcher = BatcherSort::new(vec![1, 5, 2]);
            batcher.merge();
            assert_eq!(vec![1, 2, 5], batcher.vs);
            assert_eq!(2, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(vec![2, 1, 5], k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_4() {
        {
            let mut batcher = BatcherSort::new(vec![1, 5, 2, 4]);
            batcher.merge();
            assert_eq!(vec![1, 2, 4, 5], batcher.vs);
            assert_eq!(3, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(vec![2, 5, 1, 4], k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_5() {
        let mut batcher = BatcherSort::new(vec![1, 5, 6, 2, 4]);
        batcher.merge();
        assert_eq!(vec![1, 2, 4, 5, 6], batcher.vs);
    }

    #[test]
    fn test_merge_8() {
        {
            let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 3, 4, 5]);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
            assert_eq!(9, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
        {
            let k = 2;
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], k);
            batcher.merge();
            assert_eq!(vec![1, 2], batcher.vs.split_at(k).0);
            assert_eq!(3, batcher.comparisons());
        }
        {
            let k = 4;
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], k);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4], batcher.vs.split_at(k).0);
            assert_eq!(8, batcher.comparisons());
        }
        {
            let k = 5;
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], k);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4, 5], batcher.vs.split_at(k).0);
            assert_eq!(8, batcher.comparisons());
        }
        {
            let k = 6;
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], k);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4, 5, 5], batcher.vs.split_at(k).0);
            assert_eq!(9, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_10() {
        let k = 5;
        let mut batcher = BatcherSort::new_k(vec![2, 4, 6, 8, 10, 1, 3, 5, 7, 9], k);
        batcher.merge();
        assert_eq!(vec![1, 2, 3, 4, 5], batcher.vs.split_at(k).0);
        assert_eq!(10, batcher.comparisons());
    }

    #[test]
    fn test_sort() {
        {
            let mut batcher = BatcherSort::new(vec![5, 1, 6, 2]);
            batcher.sort();
            assert_eq!(vec![1, 2, 5, 6], batcher.vs);
        }
        {
            let mut batcher = BatcherSort::new(vec![5, 4, 3, 2, 1]);
            batcher.sort();
            assert_eq!(vec![1, 2, 3, 4, 5], batcher.vs);
        }
        {
            let mut batcher = BatcherSort::new(vec![7, 6, 5, 4, 3, 2, 1]);
            batcher.sort();
            assert_eq!(vec![1, 2, 3, 4, 5, 6, 7], batcher.vs);
        }
        {
            let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 4, 3, 5]);
            batcher.sort();
            assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(vec![5, 1, 6, 7], k);
            batcher.sort();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
        }
    }

    #[quickcheck]
    fn prop_sort(xs: Vec<usize>) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }
        let mut sorted = xs.clone();
        sorted.sort();

        let mut batcher = BatcherSort::new(xs);
        batcher.sort();
        TestResult::from_bool(batcher.vs == sorted)
    }

    #[quickcheck]
    fn prop_sort_k(xs: Vec<u16>, k: usize) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }

        if k > xs.len() {
            return TestResult::discard();
        }

        let mut sorted = xs.clone();
        sorted.sort();

        let mut batcher = BatcherSort::new_k(xs, k);
        batcher.sort();

        TestResult::from_bool(batcher.vs.split_at(k).0 == sorted.split_at(k).0)
    }
}
