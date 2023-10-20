use crate::AsyncComparator;
use rayon;
use std::sync::mpsc;
use std::thread;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Task {
    v0: usize,
    v1: usize,
    level: usize,
}

impl Task {
    pub fn new(v0: usize, v1: usize, level: usize) -> Self {
        Self { v0, v1, level }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TaskState {
    Ok(Task),
    Blocked,
    Done,
}

struct TaskManager {
    // TODO consider using linked list since we're doing a lot of removes
    remaining: Vec<Task>,
    processing: Vec<Task>,
}

impl TaskManager {
    fn new(network: &[Task]) -> Self {
        Self {
            remaining: network.to_vec(),
            processing: vec![],
        }
    }

    // we need to be careful for situations such as
    // processing: (0, 1)
    // remaining: (1, 2) (2, 3)
    // since we skip (1, 2) as it conflicts with (0, 1)
    // but we should not accept (2, 3) since it depends on (1, 2)
    // To avoid this, we will only accept tasks that are 1 level higher
    // than the tasks in `self.processing`
    fn conflict_with_processing(&self, task: &Task) -> bool {
        let mut max_level = 0;
        for t in self.processing.iter() {
            if max_level < t.level {
                max_level = t.level;
            }

            // tasks on the same level should never conflict
            if task.level == t.level {
                continue;
            }

            // otherwise, the task at a higher level
            // may depend on tasks on lower levels,
            // not necessarily one level lower
            if task.v0 == t.v0 || task.v1 == t.v0 || task.v0 == t.v1 || task.v1 == t.v1 {
                return true;
            }
        }

        // the new task cannot be 2 or more levels higher than
        // the tasks in `self.processing` since we do not allow
        // our algorithm to "skip" levels, all the tasks are sorted by level
        if max_level == task.level || max_level + 1 == task.level {
            false
        } else {
            true
        }
    }

    fn remove_finished_task(&mut self, finished_task: Task) {
        let index = self
            .processing
            .iter()
            .position(|x| *x == finished_task)
            .unwrap();
        self.processing.remove(index);
    }

    /// Output the next task in the queue.
    /// There are a few steps for finding the next task when one is finished
    /// - take `finished_task` out of `self.processing`
    /// - find the next task that does not "conflict" with any current tasks
    fn next_task(&mut self, finished_task: Task) -> TaskState {
        self.remove_finished_task(finished_task);

        if self.remaining.is_empty() {
            return TaskState::Done;
        }

        // Iterate over the remaining tasks on the same level and see
        // if there is one we can execute.
        // They must be on the same level because only this
        // way we can guarantee that the execution order for these tasks
        // do not matter.
        let curr_level = self.remaining[0].level;
        for i in 0..self.remaining.len() {
            if self.remaining[i].level == curr_level
                && !self.conflict_with_processing(&self.remaining[i])
            {
                let t = self.remaining.remove(i);
                self.processing.push(t);
                return TaskState::Ok(t);
            }
        }
        return TaskState::Blocked;
    }

    /// Output the initial tasks,
    /// i.e., tasks that have no dependency at level 0
    fn initial_tasks(&mut self) -> Vec<Task> {
        // we need at least one comparator at level 0
        assert_eq!(self.remaining[0].level, 0);
        // we must be at the start of the tasks
        assert!(self.processing.is_empty());

        let mut i = 0usize;
        loop {
            if self.remaining.is_empty() {
                break;
            }
            if self.remaining[i].level == 0 {
                i += 1;
            } else {
                break;
            }
        }
        self.processing = self.remaining.drain(0..i).collect();
        self.processing.clone()
    }
}

pub fn do_work<CMP>(n_threads: usize, network: &[Task], cmp: CMP, vs: &[CMP::Item])
where
    CMP: AsyncComparator + Sync + Send + Clone,
{
    let (pool_tx, pool_rx): (mpsc::Sender<Option<Task>>, mpsc::Receiver<Option<Task>>) =
        mpsc::channel();
    let (man_tx, man_rx): (mpsc::Sender<Task>, mpsc::Receiver<Task>) = mpsc::channel();

    let mut man = TaskManager::new(network);

    // start a thread for manager
    let man_handler = thread::spawn(move || {
        // send the initial tasks
        for task in man.initial_tasks() {
            pool_tx.send(Some(task)).unwrap();
        }

        // if there are more threads than initial tasks
        // try to send those too
        if n_threads > man.processing.len() {
            // TODO
        }

        // listen to job completion and figure out the next tasks
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
mod test {
    use super::*;

    #[test]
    fn test_task_manager_basic() {
        let network = vec![Task::new(0, 1, 0), Task::new(1, 2, 1)];
        let mut man = TaskManager::new(&network);

        // check the initial task
        assert_eq!(man.initial_tasks(), vec![Task::new(0, 1, 0)]);

        // now we have one tasks in processing and it conflicts
        assert!(man.conflict_with_processing(&Task::new(1, 2, 1)));

        // suppose the next task is finished,
        // we should obtain a new task
        assert_eq!(
            TaskState::Ok(Task::new(1, 2, 1)),
            man.next_task(Task::new(0, 1, 0))
        );

        // finally we should receive done after the last task is completed
        assert_eq!(TaskState::Done, man.next_task(Task::new(1, 2, 1)));
    }
}
