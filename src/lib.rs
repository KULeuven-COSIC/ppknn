pub struct BatcherSort<T>
where
    T: Ord,
{
    pub vs: Vec<T>,
    k: usize,
    comp_count: usize,
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
        }
    }

    pub fn new_k(vs: Vec<T>, k: usize) -> Self {
        assert!(k > 0);
        Self {
            vs,
            k,
            comp_count: 0,
        }
    }

    pub fn sort(&mut self) {
        self.sort_rec(0, self.vs.len());
    }

    /*
    pub fn sort_k(&mut self) {
        self.sort_k_rec(0, self.vs.len());
    }
    */

    fn sort_rec(&mut self, start: usize, len: usize) {
        println!("[sort_rec begin] lo={}, n={}", start, len);
        if len > 1 {
            let n = len / 2;
            let m = len - n;
            self.sort_rec(start, n);
            self.sort_rec(start + n, m);

            let indices: Vec<_> = (start..start + len).collect();
            let (ix, jx) = indices.split_at(n);
            self.merge_rec(&ix, &jx);
        }
        println!("[sort_rec exit] lo={}, n={}", start, len);
    }

    /*
    fn sort_k_rec(&mut self, lo: usize, n: usize) {
        println!("[sort_k_rec begin] lo={}, n={}", lo, n);
        if n > 1 {
            let m = n / 2;
            // TODO this is broken, after first call the index
            // is wrong for the second recursive call
            self.sort_k_rec(lo, m);
            self.sort_k_rec(lo + m, m);
            self.merge_k();
        }
        println!("[sort_k_rec exit] lo={}, n={}", lo, n);
    }
    */

    /// We assume the two sorted arrays we wish to merge are consecutive of length n+m,
    /// the first one has length `n` that's always even and the second one has `m`.
    pub fn merge(&mut self) {
        let mut n = self.vs.len() / 2;
        let mut m = self.vs.len() - n;

        // make sure n is even
        if n % 2 == 1 {
            std::mem::swap(&mut n, &mut m);
        }

        let ix: Vec<_> = (0..n).collect();
        let jx: Vec<_> = (n..n + m).collect();
        self.merge_rec(&ix, &jx)
    }

    fn merge_rec(&mut self, ix: &[usize], jx: &[usize]) {
        println!("[merge begin] ix={:?}, jx={:?}", ix, jx);
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
                self.compare_at(odd_all[i], even_all[i + 1]);
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
        println!("[merge exit] ix={:?}, jx={:?}", ix, jx);
    }

    /*
    /// We assume the two arrays we wish to merge are consecutive,
    /// has length `n` and start at index `lo`.
    /// Only the first `k` elements in the two arrays
    /// are expected to be sorted.
    /// The output will only have `k` sorted elements.
    pub fn merge_k(&mut self) {
        let n = self.vs.len();
        let k = self.k;
        println!("[merge_k begin] n={}, k={}", n, k);

        // Given this kind of array,
        // [0, ..., n/2-1] [n/2, ..., n-1]
        // we transform it to
        // [0, ..., k-1] [k, ..., 2k - 1]
        // by removing elements at indices k to n/2-1 (inclusive)
        // from the first array
        // and removing indices from n-1-k to n-1
        // from the second array.
        if k < n / 2 {
            // remove the last k elements
            self.vs.truncate(n / 2 + k);
            // remove the elements at indices k to n/2-1
            let mut idx = 0;
            self.vs.retain(|_| {
                let res = idx < k || idx > n / 2 - 1;
                idx += 1;
                res
            });
            assert_eq!(self.vs.len(), k * 2);
        }

        // We truncated n, potentially, so reset it
        let n = self.vs.len();
        let lo = 0;
        let r = 1;

        let m = r * 2;
        if m < n {
            self.merge(lo, n, m);
            self.merge(lo + r, n, m);
            for i in (lo + r..lo + n - r).step_by(m) {
                if i >= k {
                    break;
                }
                self.compare_at(i, i + r);
            }
        } else {
            self.compare_at(lo, lo + r);
        }

        // keep only the `k` elements
        self.vs.truncate(k);
        println!("[merge_k exit] lo={}, n={}, r={}", lo, n, r);
    }
    */

    /// Swap in-place an elements at index `i` with another at index `j`
    fn compare_at(&mut self, i: usize, j: usize) {
        println!("[compare_at] i={}, j={}", i, j);
        self.comp_count += 1;
        if self.vs[i] > self.vs[j] {
            self.vs.swap(i, j);
        }
    }

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
        }
        {
            // let mut batcher = BatcherSort::new_k(vec![2, 1], 1);
            // batcher.merge_k();
            // assert_eq!(vec![1], batcher.vs);
        }
    }

    #[test]
    fn test_merge_3() {
        let mut batcher = BatcherSort::new(vec![1, 5, 2]);
        batcher.merge();
        assert_eq!(vec![1, 2, 5], batcher.vs);
    }

    #[test]
    fn test_merge_4() {
        {
            let mut batcher = BatcherSort::new(vec![1, 5, 2, 4]);
            batcher.merge();
            assert_eq!(vec![1, 2, 4, 5], batcher.vs);
        }
        {
            // let mut batcher = BatcherSort::new_k(vec![1, 5, 2, 4], 1);
            // batcher.merge_k();
            // assert_eq!(vec![1], batcher.vs);
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
        /*
        {
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], 1);
            batcher.merge_k();
            assert_eq!(vec![1], batcher.vs);
            assert_eq!(1, batcher.comparisons());
        }
        {
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], 2);
            batcher.merge_k();
            assert_eq!(vec![1, 2], batcher.vs);
            assert_eq!(3, batcher.comparisons());
        }
        {
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], 4);
            batcher.merge_k();
            assert_eq!(vec![1, 2, 3, 4], batcher.vs);
            assert_eq!(8, batcher.comparisons());
        }
        {
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], 5);
            batcher.merge_k();
            assert_eq!(vec![1, 2, 3, 4, 5], batcher.vs);
            assert_eq!(8, batcher.comparisons());
        }
        {
            let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], 6);
            batcher.merge_k();
            assert_eq!(vec![1, 2, 3, 4, 5, 5], batcher.vs);
            assert_eq!(9, batcher.comparisons());
        }
        */
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
        // {
        //     let mut batcher = BatcherSort::new_k(vec![5, 1, 6, 7], 1);
        //     batcher.sort_k();
        //     assert_eq!(vec![1], batcher.vs);
        // }
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
}
