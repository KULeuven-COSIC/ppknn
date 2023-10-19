use crate::AsyncComparator;
use rayon;
use std::sync::mpsc;
use std::thread;

#[derive(Copy, Clone, Debug)]
struct Task {
    v0: usize,
    v1: usize,
    level: usize,
}

#[derive(Copy, Clone, Debug)]
enum TaskState {
    Ok(Task),
    Blocked,
    Done,
}

struct TaskManager {
    remaining: Vec<Task>,
    processing: Vec<Task>,
}

impl TaskManager {
    fn new() -> Self {
        unimplemented!()
    }

    // we need to be careful for situations such as
    // processing: (0, 1)
    // remaining: (1, 2) (2, 3)
    // since we skip (1, 2) as it conflicts with (0, 1)
    // but we should not accept (2, 3) since it depends on (1, 2)
    fn conflict_with_processing(&self, task: &Task) -> bool {
        // for t in self.processing.iter() {
        //     // tasks on the same level should never conflict
        //     if task.level == t.level {
        //         continue
        //     }
        //
        //     // tasks on different levels
        // }
        // true
        unimplemented!()
    }

    /// Output the next task in the queue.
    /// There are a few steps for finding the next task when one is finished
    /// - take `finished_task` out of `self.processing`
    /// - find the next task that does not "conflict" with any current tasks
    fn next_task(&mut self, finished_task: Task) -> TaskState {
        unimplemented!()
    }

    /// Output the initial tasks,
    /// i.e., tasks that have no dependency at level 0
    fn initial_tasks(&mut self) -> Vec<Task> {
        // we need at least one comparator at level 0
        assert_eq!(self.remaining[0].level, 0);
        let mut i = 0usize;
        loop {
            if self.remaining.is_empty() {
                break;
            }
            if self.remaining[i].level == 0 {
                i += 1;
            } else {
                break
            }
        }
        self.remaining.drain(0..i).collect()
    }
}

fn do_work<CMP>(n_threads: usize, cmp: CMP, vs: &[CMP::Item])
where
    CMP: AsyncComparator + Sync + Send + Clone,
{
    let (pool_tx, pool_rx): (mpsc::Sender<Option<Task>>, mpsc::Receiver<Option<Task>>) =
        mpsc::channel();
    let (man_tx, man_rx): (mpsc::Sender<Task>, mpsc::Receiver<Task>) = mpsc::channel();

    let mut man = TaskManager::new();

    // start a thread for manager
    let man_handler = thread::spawn(move || {
        // send the initial tasks
        for task in man.initial_tasks() {
            pool_tx.send(Some(task)).unwrap();
        }

        loop {
            let finished_task = man_rx.recv().unwrap();
            match man.next_task(finished_task) {
                TaskState::Ok(task) => {
                    pool_tx.send(Some(task)).unwrap();
                }
                TaskState::Blocked => {
                    // Do nothing.
                    // Wait for the next finished_task to come in
                    // and hopefully it'll be unblocked.
                }
                TaskState::Done => {
                    pool_tx.send(None).unwrap();
                    return;
                }
            }
        }
    });

    // start the thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    pool.install(move || {
        loop {
            match pool_rx.recv().unwrap() {
                None => break,
                Some(task) => {
                    // perform the task in the thread pool
                    let man_tx = man_tx.clone();
                    let cmp = cmp.clone();
                    rayon::scope(move |s| {
                        s.spawn(move |_| {
                            // do work
                            cmp.compare(&vs[task.v0], &vs[task.v1]);
                            // send the manager a message when the task is done
                            man_tx.send(task).unwrap();
                        });
                    });
                }
            }
        }
    });

    man_handler.join().unwrap();
    // pool.join();
}

#[cfg(test)]
mod test {}
