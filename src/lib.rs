pub struct BatcherSort<T>
where
    T: Ord,
{
    pub vs: Vec<T>,
}

impl<T> BatcherSort<T>
where
    T: Ord,
{
    pub fn new(vs: Vec<T>) -> Self {
        Self { vs }
    }

    pub fn sort(&mut self) {
        self.sort_rec(0, self.vs.len());
    }

    fn sort_rec(&mut self, lo: usize, n: usize) {
        if n > 1 {
            let m = n / 2;
            self.sort_rec(lo, m);
            self.sort_rec(lo + m, m);
            self.merge_rec(lo, n, 1);
        }
    }

    /// We assume the two arrays we wish to merge are consecutive,
    /// has length `n` and start at index `lo`.
    pub fn merge_rec(&mut self, lo: usize, n: usize, r: usize) {
        let m = r * 2;
        if m < n {
            self.merge_rec(lo, n, m);
            self.merge_rec(lo + r, n, m);
            for i in (lo + r..lo + n - r).step_by(m) {
                compare_at(&mut self.vs, i, i + r);
            }
        } else {
            compare_at(&mut self.vs, lo, lo + r);
        }
    }
}

/// Swap in-place an elements at index `i` with another at index `j`
fn compare_at<T>(vs: &mut Vec<T>, i: usize, j: usize)
where
    T: Ord,
{
    if vs[i] > vs[j] {
        vs.swap(i, j);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_merge_2() {
        let mut batcher = BatcherSort::new(vec![2, 1]);
        batcher.merge_rec(0, 2, 1);
        assert_eq!(vec![1, 2], batcher.vs);
    }

    #[test]
    fn test_merge_4() {
        let mut batcher = BatcherSort::new(vec![1, 5, 2, 4]);
        batcher.merge_rec(0, 4, 1);
        assert_eq!(vec![1, 2, 4, 5], batcher.vs);
    }

    #[test]
    fn test_merge_8() {
        let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 3, 4, 5]);
        batcher.merge_rec(0, 8, 1);
        assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
    }

    #[test]
    fn test_sort() {
        let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 4, 3, 5]);
        batcher.sort();
        assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
    }
}
