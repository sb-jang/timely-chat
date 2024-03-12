import math
from collections import defaultdict
from typing import Iterator, List, Optional, TypeVar

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

T_co = TypeVar("T_co", covariant=True)


class ContextBatchedDistributedSampler(Sampler[T_co]):
    r"""
    DDP 설정에서 같은 컨텍스트끼리 한 배치로 묶을 수 있도록 커스텀한 DistributedSampler입니다.
    __iter__에서 indices 계산하는 부분만 수정하였습니다.

    :param dataset: 리워드 모델 학습용 PairwiseDataset
    :param context_indices_per_instance: 각 데이터 instance별 context index를 담고 있는 list
    :param num_replicas: DDP GPU 프로세스 (replica) 개수
    :param rank: 현재 프로세스의 랭크
    :param shuffle: 데이터셋 index들을 shuffle할지 여부
    :param seed: 랜덤 시드
    :param drop_last: 마지막 자투리를 버릴지 여부
    """

    def __init__(
        self,
        dataset: Dataset,
        context_indices_per_instance: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        # 각 context별 instance index 리스트를 계산
        self.instance_indices_per_context = defaultdict(list)
        for instance_idx, context_idx in enumerate(context_indices_per_instance):
            self.instance_indices_per_context[context_idx].append(instance_idx)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        # context별 instance index들을 담고 있는 array의 list
        indices = [np.array(instance_indices) for instance_indices in self.instance_indices_per_context.values()]

        if self.shuffle:
            # 랜덤하게 shuffle하되, 같은 컨텍스트의 데이터가 연속적으로 위치하게 함
            rng = np.random.default_rng(self.seed + self.epoch)
            # context 단위로 shuffle
            rng.shuffle(indices, axis=0)
            # context 내에서 instance index들을 shuffle
            for i in range(len(indices)):
                rng.shuffle(indices[i])

        # array of array 였던 것을 list로 flatten
        indices = np.hstack(indices).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # rank별로 학습할 index들을 차례대로 rank에 할당
        # indices에 있는 순서대로 전체 프로세스에서 학습을 진행함
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
