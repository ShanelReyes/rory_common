import hashlib as H
import time as T
import asyncio
import pytest
from rorycommon import Common as RoryCommon
import os 
import numpy as np
from option import Some
from mictlanx import AsyncClient
from mictlanx.utils import Utils
from mictlanx.utils.segmentation import Chunks,Chunk
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.clustering.secure.pqc.skmeans import Skmeans as SkmeansPQC
from rory.core.classification.secure.pqc.sknn import SecureKNearestNeighbors as SKNNPQC
from rorycommon.utils import Utils as RoryCommonUtils

MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("rory-common"))
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_ROUTERS             = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666") #mictlanx-peer-2:localhost:7002")
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",1)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_DISABLED_LOG        = bool(int(os.environ.get("MICTLANX_DISABLED_LOG",0)))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")
MICTLANX_BUCKET_ID           = os.environ.get("MICTLANX_BUCKET_ID","rory") 
MICTLANX_OUTPUT_PATH         = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")

client = AsyncClient(
    client_id=MICTLANX_CLIENT_ID,
    capacity_storage="200mb",
    debug=True,
    eviction_policy="LRU",
    max_workers= MICTLANX_MAX_WORKERS,
    routers=list(Utils.routers_from_str(routers_str=MICTLANX_ROUTERS,protocol="http")),
    verify=False
)


class MiniBatchKMeansCustom:
    def __init__(self, n_clusters, init_size=None, random_state=None):
        """
        n_clusters : int
            The number of clusters, K.
        init_size : int or None
            Number of samples to use for initializing centroids.
            If None, defaults to 10 * n_clusters.
        random_state : int or None
            Seed for reproducibility.
        """
        self.K = n_clusters
        self.init_size = init_size or 10 * n_clusters
        self.random_state = np.random.RandomState(random_state)
        self.cluster_centers_ = None  # shape (K, p)
        self.counts_ = None           # cumulative counts n_k(t), shape (K,)

    def _initialize(self, X_init):
        n_samples, _ = X_init.shape
        centers = []

        # 1. Choose the first center randomly
        idx = self.random_state.randint(n_samples)
        centers.append(X_init[idx])

        for _ in range(1, self.K):
            # 2. Compute distance from each point to nearest center
            dists = np.min(
                [np.sum((X_init - c)**2, axis=1) for c in centers],
                axis=0
            )
            # 3. Choose a new center with probability proportional to distance²
            probs = dists / np.sum(dists)
            new_idx = self.random_state.choice(n_samples, p=probs)
            centers.append(X_init[new_idx])

        self.cluster_centers_ = np.array(centers).astype(float)
        self.counts_ = np.zeros(self.K, dtype=int)
    def partial_fit(self, X_batch: np.ndarray):
        """
        Realiza una actualización incremental con un batch.
        """
        if self.cluster_centers_ is None:
            init_chunk = X_batch
            if len(init_chunk) > self.init_size:
                init_chunk = init_chunk[self.random_state.choice(len(init_chunk), self.init_size, replace=False)]
            self._initialize(init_chunk)

        # Asignar puntos al cluster más cercano
        x_sq = np.sum(X_batch**2, axis=1)[:, None]
        mu_sq = np.sum(self.cluster_centers_**2, axis=1)[None, :]
        cross = X_batch @ self.cluster_centers_.T
        dists = x_sq + mu_sq - 2 * cross
        labels = np.argmin(dists, axis=1)

        # Actualizar centroides por promedio del batch
        for k in range(self.K):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                continue
            Xk = X_batch[members]
            n_k = len(Xk)
            self.counts_[k] += n_k
            eta = 1.0 / self.counts_[k]
            mean_k = np.mean(Xk, axis=0)
            self.cluster_centers_[k] = (1 - eta) * self.cluster_centers_[k] + eta * mean_k

        return self


    def predict(self, X):
        """
        Assign each row of X to the nearest centroid.
        """
        # same distance computation as above
        x_sq = np.sum(X**2, axis=1)[:, None]
        mu_sq = np.sum(self.cluster_centers_**2, axis=1)[None, :]
        cross = X.dot(self.cluster_centers_.T)
        dists = x_sq + mu_sq - 2*cross
        return np.argmin(dists, axis=1)

    def fit(self, X_batch):
        """
        Convenience method: consume batches from batch_iterator and partial_fit each.
        batch_iterator: an iterable yielding arrays of shape (b, p)
        """
        # for X_batch in batch_iterator:
        self.partial_fit(X_batch)
        return self

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_put_chunks():
    ball_id              = "bx"
    # chunks_generator = RoryCommonUtils.read_chunks_numpy(ball_id=ball_id,filename="/rory/source/auditdatadata.npy",row_chunk=10)
    chunks_generator = RoryCommonUtils.read_chunks_numpy(ball_id=ball_id,filename="/rory/source/X.npy",row_chunk=100)
    bucket_id        = "rory"
    max_tries        = 1 
    timeout          = 120 
    for c in chunks_generator:
        res = await RoryCommon.delete_and_put_chunk(

            client    = client,
            bucket_id = bucket_id,
            ball_id    = ball_id,
            chunk      = c,
            tags      = c.metadata,
            max_tries = max_tries,
            timeout   = timeout
        )
        print(res)

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get_chunks():
    ball_id              = "bx"
    bucket_id        = "rory"
    max_tries        = 10 
    timeout          = 120 
    k = 3
    kmeans = MiniBatchKMeansCustom(n_clusters=k, random_state=None)

    async for (chunk_index, c) in RoryCommon.iterate_matrix_chunks(
        client=client,
        bucket_id=bucket_id,
        ball_id=ball_id,
        max_retries=max_tries,
        timeout=timeout
    ):
        kmeans.partial_fit(X_batch= c)
        print(kmeans.counts_)
        # print("GET SUCCESSFULLY",c[0])
        # print("_"*10)

    # RoryCommon.get
        # print(res)
        # chunk_index+=1



@pytest.mark.asyncio
async def test_segment_and_put_lazy():
    bucket_id      = "rory"
    ball_id        = "bx"
    path           = "/rory/source/X.npy"
    row_chunk_size = 100
    res = RoryCommon.segment_and_put_lazy(client= client, bucket_id=bucket_id, ball_id=ball_id, path= path, row_chunk_size= row_chunk_size)
    async for r in res:
        print(r)

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_lazy_encryption():
    dataowner = DataOwner(
        liu_scheme= Liu(
            _round         = True,
            decimals       = 2,
            secure_random  = False,
            seed           = 1,
            use_np_random  = True,
            security_level = 128
        )
    )
    ball_id              = "bx"
    chunks_generator = RoryCommonUtils.read_chunks_numpy(ball_id=ball_id,filename="/rory/source/X.npy",row_chunk=100)
    print(chunks_generator)
    for c in chunks_generator:
        maybe_ndarray = c.to_ndarray()
        if maybe_ndarray.is_some:
            res = dataowner.liu_encrypt_matrix_chunk(maybe_ndarray.unwrap())
            Chunk.from_ndarray(group_id=c.group_id, index=c.index,ndarray=res,metadata={}, chunk_id=Some(c.chunk_id))
            print(res.shape)
        del maybe_ndarray
    # for c in 