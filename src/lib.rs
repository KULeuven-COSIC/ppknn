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
        assert!(k > 0);
        Self { vs, k, comp_count: 0 }
    }

    pub fn new_k(vs: Vec<T>, k: usize) -> Self {
        assert!(k > 0);
        Self { vs, k, comp_count: 0}
    }

    pub fn sort(&mut self) {
        self.sort_rec(0, self.vs.len());
    }

    fn sort_rec(&mut self, lo: usize, n: usize) {
        if n > 1 {
            let m = n / 2;
            self.sort_rec(lo, m);
            self.sort_rec(lo + m, m);
            self.merge(lo, n, 1);
        }
    }

    /// We assume the two sorted arrays we wish to merge are consecutive,
    /// has length `n` and start at index `lo`.
    pub fn merge(&mut self, lo: usize, n: usize, r: usize) {
        let m = r * 2;
        if m < n {
            self.merge(lo, n, m);
            self.merge(lo + r, n, m);
            for i in (lo + r..lo + n - r).step_by(m) {
                self.compare_at(i, i + r);
            }
        } else {
            self.compare_at(lo, lo + r);
        }
        println!("lo={}, n={}, r={}", lo, n, r);
    }

    /// We assume the two arrays we wish to merge are consecutive,
    /// has length `n` and start at index `lo`.
    /// Only the first `k` elements in the two arrays
    /// are expected to be sorted.
    /// The output will only have `k` sorted elements.
    /// We assume `k >= n/2`.
    pub fn merge_k(&mut self, lo: usize, n: usize, r: usize) {
        let k = self.k;
        assert!(k >= n/2);
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
        // println!("lo={}, n={}, r={}", lo, n, r);
    }

    /// Swap in-place an elements at index `i` with another at index `j`
    fn compare_at(&mut self, i: usize, j: usize) {
        // println!("i={}, j={}", i, j);
        self.comp_count += 1;
        if self.vs[i] > self.vs[j] {
            self.vs.swap(i, j);
        }
    }

    pub fn comparisons(&self) -> usize {
        self.comp_count
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_merge_2() {
        let mut batcher = BatcherSort::new(vec![2, 1]);
        batcher.merge(0, 2, 1);
        assert_eq!(vec![1, 2], batcher.vs);
    }

    #[test]
    fn test_merge_4() {
        let mut batcher = BatcherSort::new(vec![1, 5, 2, 4]);
        batcher.merge(0, 4, 1);
        assert_eq!(vec![1, 2, 4, 5], batcher.vs);
    }

    #[test]
    fn test_merge_8() {
        let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 3, 4, 5]);
        batcher.merge(0, 8, 1);
        assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
        assert_eq!(9, batcher.comparisons());
    }

    #[test]
    fn test_sort() {
        let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 4, 3, 5]);
        batcher.sort();
        assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
    }

    #[test]
    fn test_merge_k_8() {
        let mut batcher = BatcherSort::new_k(vec![1, 5, 6, 7, 2, 3, 4, 5], 4);
        batcher.merge_k(0, 8, 1);
        assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
        assert_eq!(8, batcher.comparisons());
    }
}
