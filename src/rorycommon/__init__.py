import time as T
import asyncio
from mictlanx import AsyncClient
from rorycommon.utils import Utils as RoryCommonUtils
import mictlanx.interfaces as InterfaceX
from option import Option,Result,Ok,Err,Some
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
from typing import Tuple, Generator,Dict,AsyncGenerator
from mictlanx.utils.segmentation import Chunks,Chunk
from rory.core.security.dataowner import DataOwner
from rory.core.security.dataowner_paillier import DataOwner as DataOwnerPHE
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from typing import List,Awaitable
from concurrent.futures import ProcessPoolExecutor
from Pyfhel import PyCtxt
import pickle
from rory.core.security.cryptosystem.pqc.ckks import Ckks
import hashlib as H
from mictlanx.logger.log import Log



DEBUG = bool(int(os.environ.get("RORY_COMMON_DEBUG","1")))
RORY_COMMON_LOG_PATH = os.environ.get("RORY_COMMON_LOG_PATH","/mictlanx/client")
L = Log(
    name                   = __name__,
    console_handler_filter = lambda r : DEBUG,
    to_file                = False,
    path                   = RORY_COMMON_LOG_PATH
)



class Common:
    """
    Common utility functions for encryption and chunk management.
    """
    
    ckks = None 
    dataowner = None


    # Plain text
    @staticmethod 
    async def from_matrix_on_disk_to_cloud_storage(
        path:str,
        extension:str,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
    ):
        """Reads a matrix from disk and puts it into cloud storage as an ndarray."""
        try:
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                print(f"Failed to read matrix from disk: {res.unwrap_err()}")
                return res
            plaintext_matrix = res.unwrap()
            result = await Common.put_ndarray(
                client      = client,
                bucket_id   = bucket_id,
                key         = ball_id,
                matrix      = plaintext_matrix,
                tags        = tags,
                timeout     = timeout,
                max_retries = max_attempts,
            )
            if result.is_err:
                print(f"Failed to put ndarray from disk to cloud storage: {result.unwrap_err()}")
                return result
            return  result
        except Exception as e:
            return Err(e)
    
    @staticmethod
    async def from_matrix_to_cloud_storage(
        plaintext_matrix:npt.NDArray,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
    ):
        try:
            result = await Common.put_ndarray(
                client      = client,
                bucket_id   = bucket_id,
                key         = ball_id,
                matrix      = plaintext_matrix,
                tags        = tags,
                timeout     = timeout,
                max_retries = max_attempts,
            )
            if result.is_err:
                print(f"Failed to put ndarray to cloud storage: {result.unwrap_err()}")
                return Err(result.unwrap_err())
            return  result
        except Exception as e:
            return Err(e)

    @staticmethod
    async def from_cloud_storage_to_matrix(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        chunk_index:int = 0,
        backoff_factor:int = 2,
        delay:int = 1,
        max_attempts:int = 5,
        timeout:int = 120,
        force:bool = True,
        max_parallel_gets:int = 10,
        headers:Dict[str,str] = {},
        chunk_size:str = "256kb",
        http2:bool = False,
    ):
        try:
            res = await Common.get_matrix_or_error(
                client            = client,
                bucket_id         = bucket_id,
                key               = ball_id,
                backoff_factor    = backoff_factor,
                chunk_index       = chunk_index,
                chunk_size        = chunk_size,
                timeout           = timeout,
                delay             = delay,
                force             = force,
                headers           = headers,
                http2             = http2,
                max_paralell_gets = max_parallel_gets,
                max_retries       = max_attempts
            )
            # if res.is_err:
                # print(f"Failed to get matrix from cloud storage: {res.unwrap_err()}")
                # return res
            # chunk = res.unwrap()
            # matrix = chunk.to_ndarray().unwrap()
            return Ok(res)
        except Exception as e:
            # print(e)
            return Err(e)
            # return Err(e)
        
    # CKKS
    @staticmethod
    async def from_matrix_on_disk_to_cloud_storage_ckks(
        path:str,
        extension:str,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
    ):
        try: 
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                return Err(res.unwrap_err())
            plaintext_matrix = res.unwrap()
            result = await Common.segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
                ball_id            = ball_id,
                bucket_id          = bucket_id,
                client             = client,
                ctx_filename       = ctx_filename,
                key                = ball_id,
                max_retries        = max_attempts,
                n                  = plaintext_matrix.shape[0]*plaintext_matrix.shape[1],
                num_chunks         = num_chunks,
                keys_path=keys_path,
                pubkey_filename    = pubkey_filename,
                plaintext_matrix   = plaintext_matrix,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,
                tags               = tags,
                timeout            = timeout,
                decimals           = decimals,
            )
            return result
        except Exception as e:
            return Err(e)

    @staticmethod
    async def from_vector_ondisk_to_cloud_storage_ckks(
        path:str,
        extension:str,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,    
        _round:bool = False
    ):
        try:
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                return Err(res.unwrap_err())
            plaintext_matrix = res.unwrap()
            result = await Common.segment_encrypt_with_vector_ckks_and_put_chunks_with_initialized_executor(
                client             = client,
                bucket_id          = bucket_id,
                key                = ball_id,
                vector             = plaintext_matrix,
                _round             = _round,
                decimals           = decimals,
                path               = keys_path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                max_attempts       = max_attempts,
                max_workers        = num_chunks,
                tags               = tags,
                timeout            = timeout,
            )
            return result
        except Exception as e:
            return Err(e)

    @staticmethod
    async def from_matrix_to_cloud_storage_ckks(
        plaintext_matrix:npt.NDArray,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,    
        _round:bool = False
    ):
        try:
            result = await Common.segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
                ball_id            = ball_id,
                bucket_id          = bucket_id,
                client             = client,
                ctx_filename       = ctx_filename,
                key                = ball_id,
                max_retries        = max_attempts,
                n                  = plaintext_matrix.shape[0]*plaintext_matrix.shape[1],
                num_chunks         = num_chunks,
                keys_path               = keys_path,
                pubkey_filename    = pubkey_filename,
                plaintext_matrix   = plaintext_matrix,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,
                tags               = tags,
                timeout            = timeout,
                decimals           = decimals,
                _round             = _round
            )
            return result
        except Exception as e:
            return Err(e)

    
    @staticmethod
    async def from_vector_to_cloud_storage_ckks(
        vector:npt.NDArray,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,    
        _round:bool = False
    ):
        try:
            result = await Common.segment_encrypt_with_vector_ckks_and_put_chunks_with_initialized_executor(
                client             = client,
                bucket_id          = bucket_id,
                key                = ball_id,
                vector             = vector,
                _round             = _round,
                decimals           = decimals,
                path               = keys_path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                max_attempts       = max_attempts,
                max_workers        = num_chunks,
                tags               = tags,
                timeout            = timeout,
            )
            return result
        except Exception as e:
            return Err(e)

    @staticmethod
    def init_ckks_worker_context(
        path: str,
        ctx_filename: str,
        pubkey_filename: str,
        secretkey_filename: str,
        relinkey_filename: str = "",
        rotatekey_filename: str = "",
        _round: bool = False,
        decimals: int = 2
    ):
        """Runs once per worker process to load the context into RAM."""
        try:
            global ckks 
            global dataowner

            ckks= Ckks.from_pyfhel(
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename =  rotatekey_filename 
            ) 
            dataowner = DataOwnerPQC(scheme= ckks)
        except Exception as e:
            print(f"Failed to initialize CKKS context: {e}")
            raise e

    @staticmethod
    async def encrypt_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        dataowner: DataOwner,
        ndarray:npt.NDArray,
        full_shape:Tuple[int,int],
        num_chunks:int,
        max_backoff:int= 5, 
        max_attempts:int = 10,
        timeout:int=120
    ):
        """
        Encrypts a chunk of data using the provided data owner and puts it into the storage system.

        Arguments:
            client (AsyncClient): The asynchronous client to interact with the storage system.
            bucket_id (str): The ID of the bucket where the chunk will be stored.
            ball_id (str): The ID of the ball to which the chunk belongs.
            index (int): The index of the chunk within the ball.
            dataowner (DataOwner): The data owner responsible for encrypting the chunk.
            ndarray (npt.NDArray): The NumPy array representing the chunk data.
            full_shape (Tuple[int, int]): The full shape of the original data.
            num_chunks (int): The total number of chunks.
            max_backoff (int, optional): The maximum backoff time for retries. Defaults to 5.
            max_attempts (int, optional): The maximum number of attempts for retries. Defaults to 10.
            timeout (int, optional): The timeout for the operation in seconds. Defaults to 120.

        """ 
        res = dataowner.liu_encrypt_matrix_chunk(ndarray)
        m = res.shape[2]
        new_full_shape = (full_shape[0],full_shape[1], m)
        new_c = Chunk.from_ndarray(
            group_id=ball_id,
            index=index,
            ndarray=res,
            metadata={
                "full_shape":str(new_full_shape),
                "dtype":str(res.dtype),
                "shape":str(res.shape),
                "num_chunks":str(num_chunks)
            }, 
            chunk_id=Some(f"{ball_id}_{index}")
        )
        res_dp_chunk = await Common.delete_and_put_chunk(
            client      = client,
            bucket_id   = bucket_id,
            ball_id     = ball_id,
            chunk       = new_c,
            tags        = {},
            max_backoff = max_backoff,
            max_tries   = max_attempts,
            timeout     = timeout
        )
        if res_dp_chunk.is_err:
            return res_dp_chunk
        return res_dp_chunk
                # print("FAILED TO PUT", new_c.chunk_id)
    @staticmethod
    async def encrypt_ckks_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        dataowner: DataOwnerPQC,
        ndarray:npt.NDArray,
        full_shape:Tuple[int,int],
        num_chunks:int,
        max_backoff:int= 5, 
        max_attempts:int = 10,
        timeout:int=120
    ):      
            encyrpted_chunk = dataowner.ckks_encrypt_matrix_chunk(ndarray)
            data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            new_c= Chunk(
                group_id = ball_id, 
                index = index, 
                data = data, 
                chunk_id = Some("{}_{}".format(ball_id,index)),
                metadata= {
                    "full_shape":str(full_shape),
                    "num_chunks":str(num_chunks),
                }
            )
            res_dp_chunk = await Common.delete_and_put_chunk(
                client=client, 
                bucket_id=bucket_id,
                ball_id=ball_id,
                chunk=new_c,
                tags ={},
                max_backoff=max_backoff, 
                max_tries=max_attempts,
                timeout=timeout
            )
            if res_dp_chunk.is_err:
                return res_dp_chunk
            return res_dp_chunk

    @staticmethod
    async def encrypt_paillier_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        dataowner: DataOwnerPHE,
        ndarray:npt.NDArray,
        full_shape:Tuple[int,int],
        num_chunks:int,
        max_backoff:int= 5, 
        max_attempts:int = 10,
        timeout:int=120
    ):      
            encrypted_chunk = dataowner.paillier_encrypt_matrix_chunk(ndarray)
            # data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            data = pickle.dumps(encrypted_chunk)
            new_c= Chunk(
                group_id = ball_id, 
                index = index, 
                data = data, 
                chunk_id = Some("{}_{}".format(ball_id,index)),
                metadata= {
                    "full_shape":str(full_shape),
                    "num_chunks":str(num_chunks),
                    "shape":str(encrypted_chunk.shape),
                    "dtype":str(encrypted_chunk.dtype)
                }
            )
            res_dp_chunk = await Common.delete_and_put_chunk(
                client=client, 
                bucket_id=bucket_id,
                ball_id=ball_id,
                chunk=new_c,
                tags ={},
                max_backoff=max_backoff, 
                max_tries=max_attempts,
                timeout=timeout
            )
            if res_dp_chunk.is_err:
                return res_dp_chunk
            return res_dp_chunk
      

    @staticmethod
    async def get_by_chunk_index(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        timeout:int =120,
        max_attempts:int = 5,
        delay:int =1,
        backoff_factor:int =2,
        # max_backoff:int=5,
        force:bool =True,
        max_parallel_gets:int =10, 
        headers:Dict[str,str]= {}, 
        chunk_size:str="256kb",
        http2:bool = False,
        max_backoff:int =3

    ):
        i =0
        while i <= max_attempts :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_parallel_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_attempts,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_attempts})")
                await asyncio.sleep(delay)
                i+=1
                continue
            return x.unwrap()
        
    @staticmethod
    async def segment_and_put_lazy(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        path:str,
        row_chunk_size:int =100,
        max_attempts:int = 10,
        timeout:int=120,max_backoff:int =5,tags:Dict[str,str]={}
    )->AsyncGenerator[InterfaceX.PutChunkedResponse, None]:
        chunks_generator = RoryCommonUtils.read_chunks_numpy(ball_id=ball_id,filename=path,row_chunk=row_chunk_size)
        for c in chunks_generator:
            res = await Common.delete_and_put_chunk(

                client    = client,
                bucket_id = bucket_id,
                ball_id    = ball_id,
                chunk      = c,
                tags      = {**c.metadata,**tags},
                max_tries = max_attempts,
                timeout   = timeout,
                max_backoff=max_backoff 
            )
            if res.is_err:
                raise Exception(f"Failed to put a chunk: {c.chunk_id}")
            yield res.unwrap()

    @staticmethod
    async def read_numpy_from(path:str="",extension:str="")->Result[npt.NDArray,Exception]:
        try:
            if extension == "csv":
                plaintextMatrix = pd.read_csv(
                    path, 
                    header=None
                ).values
                return Ok(plaintextMatrix)
            elif extension == "npy":
                with open(path, "rb") as f:
                    plaintextMatrix = np.load(f)
                    return Ok(plaintextMatrix.astype(np.float64))
            else:
                return Err(Exception("Either path or extension  was not provided"))
        except Exception as e:
            return Err(e)
    
    # Serializer
    @staticmethod
    def from_pyctxt_list_to_bytes(xs:List[PyCtxt]):
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in xs]
        return pickle.dumps(serialized_ciphertexts)
    
    def from_pyctxt_matrix_to_bytes(xs:List[PyCtxt]):
        serialized_ciphertexts = []
        for ctxts in xs:
            inner_sctxts = []
            for ctxt in ctxts:
                inner_sctxts.append(ctxt.to_bytes())
            serialized_ciphertexts.append(inner_sctxts)
        return pickle.dumps(serialized_ciphertexts)
    
    def from_bytes_to_pyctxt_matrix(ckks:Ckks,x:bytes):
        yss = pickle.loads(x)
        scheme = ckks.he_object
        result = []
        for ys in yss: 
            tmp_row = []
            for y in ys:
                _y = PyCtxt(None,scheme,None,y,'FRACTIONAL')
                tmp_row.append(_y)
            result.append(tmp_row)
        _res = np.vstack(result)
        return _res
    

    @staticmethod
    def from_bytes_to_pyctxt_list_v1(ckks:Ckks,x:bytes)->List[PyCtxt]:
        scheme  = ckks.he_object
        xx      = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), x))
        return xx


    @staticmethod
    def from_pyctxts_to_chunks(key:str,xs:List[PyCtxt],num_chunks:int=2)->Option[Chunks]:
        try:
            n = len(xs)
            chs = Chunks._iter_to_chunks(num_chunks=num_chunks,chunk_prefix=Some(key),group_id=key,n=n,iterable=xs)
            def __inner():
                for c in chs: 
                    chunk_id = Some(c.get("chunk_id",None)).filter(lambda x: not x == None)
                    data = [x.to_bytes() for x in c["data"]]
                    c_tmp = Chunk.from_list(
                        group_id = c["group_id"],
                        index    = c["index"],
                        chunk_id = chunk_id,
                        metadata = c["metadata"],
                        xs       = data,
                    )
                    yield c_tmp
            return Some(Chunks(chs= __inner() , n = n ))
        except Exception as e:
            print(e)
            return e
    

    @staticmethod
    def from_pyctxt_matrix_to_chunks(key:str,xs:List[List[PyCtxt]],num_chunks:int=2)->Option[Chunks]:
        try:
            n = len(xs)
            chs = Chunks._iter_to_chunks(num_chunks=num_chunks,chunk_prefix=Some(key),group_id=key,n=n,iterable=xs)
            def __inner():
                for c in chs:
                    chunk_id = Some(c.get("chunk_id",None)).filter(lambda x: not x == None)
                    ys    = c["data"]
                    data  = Common.from_pyctxt_matrix_to_bytes(ys)
                    c_tmp = Chunk(
                        group_id = c["group_id"],
                        index    = c["index"],
                        chunk_id = chunk_id,
                        data     = data
                    )
                    yield c_tmp
            return Some(Chunks(chs= __inner() , n = n ))
        except Exception as e:
            print(e)
            return e
        

    @staticmethod
    def from_chunks_to_pyctxts_list(ckks:Ckks, chunks:Chunks)->List[PyCtxt]:
        chunks.sort()
        xs = []
        for c in chunks:
            x = c.to_list().unwrap()
            xx = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            xs.extend(xx)
        return xs


    @staticmethod
    def from_bytes_to_pyctxt_list(ckks:Ckks,xs:List[bytes])->List[PyCtxt]:
        scheme  = ckks.he_object
        xx = []
        for x in xs:
            y = PyCtxt(None, scheme, None,x, "FRACTIONAL")
            xx.append(y)
        return xx
    

    @staticmethod
    def from_bytes_to_pyctxt_list_v2(ckks:Ckks, x:bytes):
        xs = pickle.loads(x)
        scheme = ckks.he_object
        xx = [PyCtxt(None, scheme, None, x, "FRACTIONAL") for x in xs ]
        return xx


    @staticmethod
    def encrypt_chunk_liu(key:str,dataowner:DataOwner,chunk:Chunk, np_random:bool)-> Chunk:
        ptm = chunk.to_ndarray().unwrap()
        encyrpted_chunk:npt.NDArray = dataowner.liu_encrypt_matrix_chunk(plaintext_matrix = ptm, np_random=np_random)
        return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray= encyrpted_chunk, chunk_id=Some("{}_{}".format(key,chunk.index)))


    @staticmethod
    def to_chunks_generator(awaitable_chunks:List[Awaitable[Chunk]]):
        xs = list(map(lambda fut: fut.result(), awaitable_chunks))
        return xs


    #  Segmentation
    @staticmethod
    def segment_and_encrypt_liu(key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2,max_workers:int = int(os.cpu_count()/2)):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
                awaitable_chunks.append(future)
            return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )


    @staticmethod
    def segment_and_encrypt_liu_with_executor(executor:ProcessPoolExecutor,key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2 ):
        plaintext_matrix_chunks:Chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    
    @staticmethod
    def segment_and_encrypt_fdhope(algorithm:str, key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int ,num_chunks:int=2, threshold:float = 0.0, max_workers:int = int(os.cpu_count()/2) ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_fdhope, key = key, dataowner = dataowner, chunk = plaintext_matrix_chunk, algorithm = algorithm, threshold = threshold)
                awaitable_chunks.append(future)
            return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )


    @staticmethod
    def segment_and_encrypt_fdhope_with_executor(executor:ProcessPoolExecutor,algorithm:str, key:str,dataowner:DataOwner,matrix:npt.NDArray, n:int ,num_chunks:int=2, sens:float = 0.00001 ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_fdhope, key = key, dataowner = dataowner, chunk = plaintext_matrix_chunk, algorithm = algorithm, sens = sens)
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)


    @staticmethod
    def encrypt_chunk_fdhope(key:str,dataowner:DataOwner,chunk:Chunk,algorithm:str,sens:float=0.00001)-> Chunk:
        try:
            encyrpted_chunk = dataowner.encrypt_udm_chunks(
                plaintext_matrix = chunk.to_ndarray().unwrap(),
                algorithm        = algorithm,
                sens             = sens
                )
            return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray= encyrpted_chunk.matrix, chunk_id=Some("{}_{}".format(key,chunk.index)))
        except Exception as e:
            print("ERROR", e)
            raise e
    

    @staticmethod
    def chunks_to_bytes_gen(chs:Chunks) -> Generator[bytes,None,None]:
        for chunk in chs.iter():
            yield chunk.data


    @staticmethod
    def encrypt_vector_ckks_with_executor(executor:ProcessPoolExecutor, key:str, vector:npt.NDArray, _round:bool, decimals:int, path:str, ctx_filename:str, pubkey_filename:str, secretkey_filename:str, relinkey_filename:str="", rotatekey_filename:str=""):
        return Common.segment_and_encrypt_ckks_with_executor(
            executor           = executor,
            key                = key,
            plaintext_matrix   = vector,
            n                  = 1,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            num_chunks         = 1,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename
        )
    
    
    @staticmethod
    def segment_and_encrypt_ckks_with_initialized_executor(
        key:str, 
        plaintext_matrix:npt.NDArray,
        n:int, 
        _round:bool,
        decimals:int,
        path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        num_chunks:int=2,
        relinkey_filename:str="",
        rotatekey_filename:str=""
    ):
        executor = ProcessPoolExecutor(
            max_workers = num_chunks,
            initializer = Common.init_ckks_worker_context,
            initargs    = (path, ctx_filename, pubkey_filename, secretkey_filename, relinkey_filename, rotatekey_filename, _round, decimals)
        )   
        plaintext_matrix_chunks = Chunks.from_ndarray( ndarray = plaintext_matrix, group_id = key, num_chunks = num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_ckks_with_initialized_executor,
                    key      = key,
                    chunk    = plaintext_matrix_chunk,
                )
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
    @staticmethod
    async def segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        key:str, 
        plaintext_matrix:npt.NDArray,
        keys_path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        n:int, 
        _round:bool= False,
        num_chunks:int = 2, 
        decimals:int = 2,
        relinkey_filename:str="",
        rotatekey_filename:str="",
        timeout:int = 120,
        max_retries:int = 5,
        tags:Dict[str,str] = {}
    ):
        encrypted_chunks = Common.segment_and_encrypt_ckks_with_initialized_executor(
            key                = key,
            plaintext_matrix   = plaintext_matrix,
            n                  = n,
            _round             = _round,
            decimals           = decimals,
            path               = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename, 
            num_chunks         = num_chunks,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            secretkey_filename = secretkey_filename
        )
        put_result = await Common.put_chunks(
            client    = client,
            bucket_id = bucket_id,
            key       = ball_id,
            chunks    = encrypted_chunks,
            max_retries=max_retries, 
            timeout=timeout,
            tags      = tags,
        )
        return put_result
    



    @staticmethod
    def segment_and_encrypt_ckks_with_executor(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool,
        decimals:int,
        path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        num_chunks:int=2,
        relinkey_filename:str="",
        rotatekey_filename:str=""
    ):
        plaintext_matrix_chunks = Chunks.from_ndarray( ndarray = plaintext_matrix, group_id = key, num_chunks = num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_ckks,
                key                = key,
                chunk              = plaintext_matrix_chunk,
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename =  rotatekey_filename,
            )
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    



    @staticmethod
    def encrypt_chunk_ckks(key:str, chunk:Chunk, _round:bool, decimals:int, path:str, ctx_filename:str, pubkey_filename:str, secretkey_filename:str, relinkey_filename:str = "", rotatekey_filename:str = "")-> Chunk:
        try:
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round             = _round,
                    decimals           = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    secretkey_filename = secretkey_filename,
                    relinkey_filename  = relinkey_filename,
                    rotatekey_filename =  rotatekey_filename 

                ) 
            )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = dataowner.ckks_encrypt_matrix_chunk(plaintext_matrix = plaintext_matrix)
            data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            c= Chunk(group_id = key, index = chunk.index, data = data, chunk_id = Some("{}_{}".format(key,chunk.index)))
            return c
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
    


    @staticmethod
    def encrypt_chunk_ckks_with_initialized_executor(key:str, chunk:Chunk)-> Chunk:
        try:
            global ckks
            global dataowner
            if ckks is None or dataowner is None:
                raise Exception("CKKS context or dataowner not initialized. Please run init_ckks_worker_context first.")
  
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = dataowner.ckks_encrypt_matrix_chunk(plaintext_matrix = plaintext_matrix)
            data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            c= Chunk(group_id = key, index = chunk.index, data = data, chunk_id = Some("{}_{}".format(key,chunk.index)))
            return c
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
            raise e
    

    @staticmethod
    def segment_and_encrypt_ckks_with_executor_v2(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool, 
        decimals:int, 
        path:str, 
        ctx_filename:str, 
        pubkey_filename:str, 
        secretkey_filename:str,
        num_chunks:int=2, 
        relinkey_filename:str="",
        rotatekey_filename:str=""
    ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray = plaintext_matrix, group_id = key, num_chunks = num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_ckks_v2,
                key                = key,
                chunk              = plaintext_matrix_chunk,
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename
            )
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)


    @staticmethod
    def encrypt_chunk_ckks_v2(
        key:str, 
        chunk:Chunk, 
        _round:bool, 
        decimals:int, 
        path:str, 
        ctx_filename:str, 
        pubkey_filename:str, 
        secretkey_filename:str, 
        relinkey_filename:str = "",
        rotatekey_filename:str = ""
        )-> Chunk:
        try:
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round             = _round,
                    decimals           = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    secretkey_filename = secretkey_filename,
                    relinkey_filename  = relinkey_filename,
                    rotatekey_filename = rotatekey_filename
                ) 
            )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[List[PyCtxt]] = dataowner.ckks_encrypt_matrix_list_chunk(plaintext_chunk = plaintext_matrix)
            data = Common.from_pyctxt_matrix_to_bytes(xs=encyrpted_chunk)
            return Chunk(group_id = key, index = chunk.index, data = data, chunk_id = Some("{}_{}".format(key,chunk.index)))
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
            raise e
    

    @staticmethod
    def from_chunks_to_pyctxt_list(chunks:Chunks, ckks:Ckks)->List[List[PyCtxt]]:
        xs = []
        for ch in chunks.iter():
            x  = pickle.loads(ch.data)
            xx = Common.from_bytes_to_pyctxt_list(ckks=ckks,xs=x)
            xs.append(xx)
        return xs
    

    def verify_mean_error(old_matrix:npt.NDArray, new_matrix:npt.NDArray, min_error:float=0.15):
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        return mean_error <= min_error


    @staticmethod
    def from_chunks_to_pyctxt_matrix(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            x = ch.data
            x  = pickle.loads(x)
            xx = Common.from_list_bytes_to_pyctxt_matrix(ckks=ckks,xs=x)
            xs.extend(xx)
        return xs
    

    @staticmethod
    def from_list_bytes_to_pyctxt_matrix(ckks:Ckks,xs:List[bytes])->List[PyCtxt]:
        scheme  = ckks.he_object
        matrix = []
        for xs in xs:
            tmp_row = []
            for x in xs:
                element = PyCtxt(None, scheme, None,x, "FRACTIONAL")
                tmp_row.append(element)
            matrix.append(tmp_row)
        return matrix
    @staticmethod
    async def while_not_delete_key(client:AsyncClient ,bucket_id:str, key:str,timeout:int = 3600,max_tries:int = 5): 
        n_deletes = -1
        i = 0
        while (n_deletes ==-1 or n_deletes >0) and i <= max_tries:
            _delete_result = await client.delete_by_key(bucket_id=bucket_id,key=key,timeout=timeout,force = True)

            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
                L.debug({
                    "event":"WHILE.NOT.DELETE.KEY.SUCCESS",
                    "bucket_id":bucket_id,
                    "ball_id":key,
                    "n_deletes":n_deletes,
                    "i":i, 
                    "max_tries":max_tries,
                    "ok":_delete_result.is_ok
                 })
                if n_deletes == 0:
                    return n_deletes
            else:
                L.error({
                    "error":str(_delete_result.unwrap_err()),
                    "bucket_id":bucket_id,
                    "ball_id":key,
                    "i":i,
                    "max_tries":max_tries,
                    "n_deletes":n_deletes,

                })
            i+=1
        return n_deletes

    @staticmethod
    async def while_not_delete_ball_id(STORAGE_CLIENT:AsyncClient ,bucket_id:str, key:str,timeout:int = 3600,max_tries:int = 5): 
        n_deletes = -1
        i = 0
        while (n_deletes ==-1 or n_deletes >0) and i <= max_tries:
            _delete_result = await STORAGE_CLIENT.delete(bucket_id=bucket_id,ball_id=key,timeout=timeout,force = True)

            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
                L.debug({
                    "event":"WHILE.NOT.DELETE.BALL_ID.SUCCESS",
                    "bucket_id":bucket_id,
                    "ball_id":key,
                    "n_deletes":n_deletes,
                    "i":i, 
                    "max_tries":max_tries,
                    "ok":_delete_result.is_ok
                 })
                if n_deletes == 0:
                    return n_deletes
            else:
                L.error({
                    "error":str(_delete_result.unwrap_err()),
                    "bucket_id":bucket_id,
                    "ball_id":key,
                    "i":i,
                    "max_tries":max_tries,
                    "n_deletes":n_deletes,

                })
            i+=1
        return n_deletes
    

    @staticmethod
    async def delete_and_put_bytes(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        data:bytes, 
        chunk_size:str="128kb",
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5
    )->Result[bool,Exception]:
        condition = True
        put_res = None
        i = 0
        while  i < max_tries: 
            _delete_result = await Common.while_not_delete_ball_id( STORAGE_CLIENT = client, bucket_id = bucket_id, key = key,max_tries=max_tries)
            put_res = await client.put(bucket_id = bucket_id, key = key, value = data, chunk_size = chunk_size, tags = tags, timeout = timeout)
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
            if condition:
                L.error({
                    "error":str(put_res.unwrap_err())
                })
                print(f"Put failed reytring in 1 second... Attemp {i+1}/{max_tries}")
                await asyncio.sleep(1)
            i+=1
            i+=1
        return put_res
   
    @staticmethod
    async def delete_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        chunk:Chunk, 
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5,
        max_backoff:int =5
    )->Result[bool,Exception]:
        condition = True
        put_res = None
        i = 0
        while  i < max_tries: 
            _delete_result = await Common.while_not_delete_key(client = client, bucket_id = bucket_id, key = chunk.chunk_id,timeout=timeout,max_tries=max_tries)
            
            put_res = await client.put_single_chunk(
                bucket_id   = bucket_id,
                ball_id     = ball_id,
                chunk       = chunk,
                tags        = tags,
                timeout     = timeout,
                max_tries   = max_tries,
                max_backoff = max_backoff
            )

            L.debug({
                "event":"DELETE.COMPLETED",
                "bucket_id":bucket_id,
                "key":ball_id,
                "n_deletes":_delete_result,
                "put_ok": put_res.is_ok
            })
            if put_res.is_ok:
                return put_res
            
            # condition = put_res.is_err or _delete_result >0
            condition = put_res.is_err and not (_delete_result == 0)
            # and not (_delete_result == 0)
            if condition:
                L.error({
                    "error":str(put_res.unwrap_err()),
                    "i":i
                })
                print(f"Put failed reytring in 1 second... Attemp {i+1}/{max_tries}")
                await asyncio.sleep(1)
            i+=1
        L.debug(
            {
                "event":"DELETE.PUT.CHUNKS",
                "bucket_id":bucket_id,
                "key":ball_id,
                "ok":put_res.is_ok
            }
        )
        return put_res

    @staticmethod
    async def delete_and_put_chunks(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        chunks:Chunk, 
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5
    )->Result[InterfaceX.PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        i = 0
        while  i < max_tries: 
            _delete_result = await Common.while_not_delete_ball_id(STORAGE_CLIENT = client, bucket_id = bucket_id, key = key,timeout=timeout,max_tries=max_tries)

            put_res = await client.put_chunks(bucket_id = bucket_id, key = key, chunks = chunks, tags = tags, timeout = timeout)
            L.debug({
                "event":"DELETE.COMPLETED",
                "bucket_id":bucket_id,
                "key":key,
                "n_deletes":_delete_result,
                "put_ok": put_res.is_ok
            })
            if put_res.is_ok:
                return put_res
            
            # condition = put_res.is_err or _delete_result >0
            condition = put_res.is_err and not (_delete_result <=0)
            # and not (_delete_result == 0)
            if condition:
                L.error({
                    "error":str(put_res.unwrap_err()),
                    "i":i
                })
                print(f"Put failed reytring in 1 second... Attemp {i+1}/{max_tries}")
                i+=1
                await asyncio.sleep(1)
                continue

        L.debug(
            {
                "event":"DELETE.PUT.CHUNKS",
                "bucket_id":bucket_id,
                "key":key,
                "ok":put_res.is_ok
            }
        )
        return put_res


    @staticmethod
    async def put_ndarray(client:AsyncClient,key:str,matrix:npt.NDArray,timeout:int =300,max_retries:int=5,tags:Dict[str,str]={},bucket_id:str= "rory"):
        put_chunks_generator_results = await Common.delete_and_put_bytes(
            client    = client,
            bucket_id = bucket_id,
            key       = key,
            data      = matrix.tobytes(),
            tags      = {
                "shape": str(matrix.shape),
                "dtype": str(matrix.dtype),
                **tags
            },
            timeout=timeout,
            max_tries=max_retries
        )
        return put_chunks_generator_results


    @staticmethod
    async def segment_and_encrypt_liu_and_put_chunks(
        executor:ProcessPoolExecutor,
        dataowner:DataOwner,
        n:int,
        np_random:bool,
        client:AsyncClient,
        key:str,matrix:npt.NDArray,
        num_chunks:int=2,
        tags:Dict[str,str]={},
        bucket_id:str= "rory"
    )->Tuple[Result[InterfaceX.PutChunkedResponse,Exception],float,float]:
        """This is only a convenience method that runs the segment_encrypt_and_putchunks and then puts the chunks. You can run them separately if you want more control over the process or want to do some processing in between."""
        return await Common.segment_encrypt_and_put_chunks(
            bucket_id=bucket_id,
            client=client,
            dataowner=dataowner,
            key=key,
            matrix=matrix,
            n=n,
            np_random=np_random,
            num_chunks=num_chunks,
            tags=tags,
            executor=executor
        )


    @staticmethod
    async def segment_encrypt_and_put_chunks(
        executor:ProcessPoolExecutor,
        dataowner:DataOwner,
        n:int,
        np_random:bool,
        client:AsyncClient,
        key:str,matrix:npt.NDArray,
        num_chunks:int=2,
        tags:Dict[str,str]={},
        bucket_id:str= "rory"
    )->Tuple[Result[InterfaceX.PutChunkedResponse,Exception],float,float]:
        t1     = T.time()
        chunks = Common.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = key,
            plaintext_matrix = matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            np_random        = np_random
        )
        seg_encrypt_rt = T.time() - t1
        put_chunks_generator_results = await Common.put_chunks(client = client, bucket_id = bucket_id, key = key, chunks = chunks, tags = tags)
        return put_chunks_generator_results,seg_encrypt_rt,T.time()-t1
    

    @staticmethod
    async def segment_encrypt_with_vector_ckks_and_put_chunks_with_executor(
        client: AsyncClient,
        bucket_id:str,
        executor:ProcessPoolExecutor, 
        key:str, 
        vector:npt.NDArray, 
        _round:bool,
        decimals:int,
        path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        relinkey_filename:str="",
        rotatekey_filename:str="",
        tags:Dict[str,str]={},
        timeout:int =300,
        max_attempts:int =5
    ) -> Result[InterfaceX.PutChunkedResponse,Exception]:
        t1     = T.time()
        

        chunks = Common.encrypt_vector_ckks_with_executor(
            executor           = executor,
            key                = key,
            vector             = vector,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename 
        )
        put_result = await Common.delete_and_put_chunks(
            client    = client,
            bucket_id = bucket_id,
            key       = key,
            chunks    = chunks,
            timeout   = timeout,
            max_tries = max_attempts,
            tags      = tags
            
        )
        return put_result
    
    @staticmethod
    async def segment_encrypt_with_vector_ckks_and_put_chunks_with_initialized_executor(
        client: AsyncClient,
        bucket_id:str,
        key:str, 
        vector:npt.NDArray, 
        _round:bool,
        decimals:int,
        path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        relinkey_filename:str="",
        rotatekey_filename:str="",
        tags:Dict[str,str]={},
        timeout:int =300,
        max_workers:int = 2,
        max_attempts:int =5
    ):
        try:
            executor = ProcessPoolExecutor(
                max_workers = max_workers,
                initializer = Common.init_ckks_worker_context,
                initargs    = (path, ctx_filename, pubkey_filename, secretkey_filename, relinkey_filename, rotatekey_filename, _round, decimals)
            )
            return await Common.segment_encrypt_with_vector_ckks_and_put_chunks_with_executor(
                client             = client,
                bucket_id          = bucket_id,
                executor           = executor,
                key                = key,
                vector             = vector,
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                tags               = tags,
                timeout            = timeout,
                max_attempts        = max_attempts

            )
        except Exception as e:
            raise e

    @staticmethod
    async def put_chunks(client:AsyncClient,key:str,chunks:Chunks,timeout:int=300,max_retries:int=5,tags:Dict[str,str]={},bucket_id:str= "rory"):
        chunks.sort()
        put_chunks_generator_results = await Common.delete_and_put_chunks(
            client    = client,
            bucket_id = bucket_id,
            key       = key,
            chunks    = chunks,
            tags      = tags,
             timeout=timeout,
             max_tries=max_retries
        )
        return put_chunks_generator_results
    
    @staticmethod
    async def get_matrix_chunk_or_error(
        client:AsyncClient,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Tuple[npt.NDArray,InterfaceX.Metadata]:
        i =0
        while i <= max_retries :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_paralell_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            (data,metadata) = x.unwrap()
            maybe_ndarray = data.to_ndarray()
            if maybe_ndarray.is_none:
                raise Exception("Failed to convert chunk into a numpy array")
            return (maybe_ndarray.unwrap(), metadata)
            # dtype = metadata.tags.get("dtype","float32")
            # raw_ndarray = np.frombuffer(buffer=data.to,dtype=dtype)
            # shape = eval(metadata.tags.get("shape",str(raw_ndarray.shape)))
            # return raw_ndarray.reshape(shape)
        raise Exception(f"Get {bucket_id}@{ball_id} failed: Max tries reached")
    
    @staticmethod
    async def get_matrix_or_error(
        client:AsyncClient,
        key:str, 
        bucket_id:str,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0,
    )->npt.NDArray:
        i =0
        while i <= max_retries :
            x = await client.get(
                bucket_id         = bucket_id,
                key               = key,
                backoff_factor    = backoff_factor,
                max_paralell_gets = max_paralell_gets,
                chunk_size        = chunk_size,
                delay             = delay,
                force             = force, 
                headers           = headers,
                http2             = http2,
                max_retries       = max_retries,
                chunk_index       = chunk_index
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            get_response = x.unwrap()
            dtype = get_response.metadatas[0].tags.get("dtype","float32")
            raw_ndarray = np.frombuffer(buffer=get_response.data.tobytes(),dtype=dtype)
            shape = eval(get_response.metadatas[0].tags.get("shape",str(raw_ndarray.shape)))
            return raw_ndarray.reshape(shape)
        raise Exception(f"Get {bucket_id}@{key} failed: Max tries reached")
    

    @staticmethod
    async def get_and_merge(
        client:AsyncClient,
        key:str,
        bucket_id:str="rory",
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0,
    ):
        try:
            i= 0
            while i < max_retries:
                x_result = client.get_chunks(
                    bucket_id         = bucket_id,
                    key               = key,
                    timeout           = timeout,
                    max_retries       = max_retries,
                    delay             = delay,
                    backoff_factor    = backoff_factor,
                    force             = force,
                    max_parallel_gets = max_paralell_gets,
                    chunk_size        = chunk_size,
                    headers           = headers,
                    http2             = http2,
                    chunk_index       = chunk_index
                )
                ms:List[InterfaceX.Metadata] = []
                xs:List[Tuple[int, npt.NDArray,bytes]] = []
                h = H.sha256()
                async for (m,data) in x_result:
                    data_bytes = data.tobytes()
                    ms.append(m)
                    shape = eval(m.tags.get("shape"))
                    dtype = m.tags.get("dtype","float64")
                    x = np.frombuffer(data_bytes,dtype= dtype ).reshape(shape)
                    index = int(m.tags.get("index","-1"))
                    xs.append((index,x,data_bytes))
                if len(ms) >0:
                    m = ms[0]
                    num_chunks = int(m.tags.get("num_chunks","-1"))
                    if num_chunks == -1 or num_chunks != len(ms):
                        raise Exception("Faile to get the chunks")
                    else:
                        full_shape  = m.tags.get("full_shape")
                        full_dtype  = m.tags.get("full_dtype","float64")
                        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
                        ordered_chunks = []
                        for (i,nd, data) in xs_sorted:
                            h2 = H.sha256()
                            h.update(data)
                            h2.update(data)
                            ordered_chunks.append(nd)
                        checksum = h.hexdigest()
                        return np.concatenate(ordered_chunks, axis=0)
        except Exception as e:
            raise e
    
    @staticmethod
    async def get_and_merge_safe(
             client:AsyncClient,
        key:str,
        bucket_id:str="rory",
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0,
    ):
        try:
            matrix = await Common.get_and_merge(
                client = client,
                key = key,
                bucket_id = bucket_id,
                max_retries = max_retries,
                delay = delay,
                backoff_factor = backoff_factor,
                max_paralell_gets = max_paralell_gets,
                force = force,
                timeout = timeout,
                chunk_size = chunk_size,
                headers = headers,
                http2 = http2,
                chunk_index = chunk_index
            )
            return Ok(matrix)
        except Exception as e:
            return Err(e) 

  
    @staticmethod
    async def iterate_matrix_chunks(
        client:AsyncClient,
        ball_id:str,
        bucket_id:str="rory",
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0
    ):

            x_result = client.get_chunks(
                bucket_id         = bucket_id,
                key               = ball_id,
                timeout           = timeout,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force,
                max_parallel_gets = max_paralell_gets,
                chunk_size        = chunk_size,
                headers           = headers,
                http2             = http2,
                chunk_index       = chunk_index
            )
            async for (m,data) in x_result:
                shape = eval(m.tags.get("shape"))
                dtype = m.tags.get("dtype","float64")
                index = int(m.tags.get("index","-1"))
                # print(shape,dtype,index)
                yield index,m,np.frombuffer(data.tobytes(),dtype=dtype).reshape(shape)

    @staticmethod
    async def get_pyctxt_chunk_or_error(
        client:AsyncClient,
        ckks:Ckks,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Tuple[List[PyCtxt],InterfaceX.Metadata]:
        i =0
        while i <= max_retries :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_paralell_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            (data,m) = x.unwrap()
            data_bytes = data.data
            x          = pickle.loads(data_bytes)
            xx         = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            return xx,m
    
        raise Exception(f"Get {bucket_id}@{ball_id} failed: Max tries reached")
    
    @staticmethod
    async def get_pyctxt_chunk(
        client:AsyncClient,
        ckks:Ckks,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Result[Tuple[List[PyCtxt],InterfaceX.Metadata],Exception]:
        try:
            x = await Common.get_pyctxt_chunk_or_error(
                client = client,
                ckks   = ckks,
                ball_id = ball_id,
                bucket_id = bucket_id,
                index = index,
                max_retries = max_retries,
                delay = delay,
                backoff_factor = backoff_factor,
                max_paralell_gets = max_paralell_gets,
                force = force,
                timeout = timeout,
                chunk_size = chunk_size,
                headers = headers,
                http2 = http2,
                max_backoff = max_backoff
            )
            return Ok(x)
        except Exception as e:
            return Err(e)
        
    
    
    @staticmethod
    async def get_paillier_chunk_or_error(
        client:AsyncClient,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Tuple[List[PyCtxt],InterfaceX.Metadata]:
        i =0
        while i <= max_retries :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_paralell_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            (data,m) = x.unwrap()
            data_bytes = data.data
            x          = pickle.loads(data_bytes)
            # xx         = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            return x,m
    
        raise Exception(f"Get {bucket_id}@{ball_id} failed: Max tries reached")
    @staticmethod
    async def get_paillier_matrix(
        client:AsyncClient,
        ball_id:str, 
        bucket_id:str,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0
    ):
        try:
            i= 0
            while i < max_retries:
                x_result = client.get_chunks(
                    bucket_id         = bucket_id,
                    key               = ball_id,
                    timeout           = timeout,
                    max_retries       = max_retries,
                    delay             = delay,
                    backoff_factor    = backoff_factor,
                    force             = force,
                    max_parallel_gets = max_paralell_gets,
                    chunk_size        = chunk_size,
                    headers           = headers,
                    http2             = http2,
                    chunk_index       = chunk_index
                )
                ms:List[InterfaceX.Metadata] = []
                xs:List[Tuple[int, npt.NDArray]] = []
                async for (m,data) in x_result:
                    ms.append(m)
                    data_bytes = data.tobytes()
                    shape      = eval(m.tags.get("shape"))
                    dtype      = m.tags.get("dtype","float64")
                    x          = pickle.loads(data_bytes)
                    index = int(m.tags.get("index","-1"))
                    xs.append((index,x))
                if len(ms) >0:
                    m = ms[0]
                    num_chunks = int(m.tags.get("num_chunks","-1"))
                    if num_chunks == -1 or num_chunks != len(ms):
                        return Err(Exception("Faile to get the chunks"))
                    else:
                        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
                        ordered_chunks = []
                        for (i,nd) in xs_sorted:
                            ordered_chunks.append(nd)
                        return Ok(np.concatenate(ordered_chunks, axis=0))
                return Err(Exception("Failed to get chunks. "))
        except Exception as e:
            return Err(e)
        

    @staticmethod
    async def get_pyctxt(
            client:AsyncClient,
            bucket_id:str, 
            key:str,
            ckks:Ckks,
            max_retries:int = 5,
            delay:float = 1,
            backoff_factor:float =.5,
            max_paralell_gets:int = 10, 
            force:bool = False,
            timeout:int = 120,
            chunk_size:str="256kb",
            headers:Dict[str,str] ={},
            http2:bool = False,
            chunk_index:int = 0
    )-> List[PyCtxt]:
        get_chunks_generator = client.get_chunks(
            key               = key,
            bucket_id         = bucket_id,
            max_retries       = max_retries,
            delay             = delay,
            backoff_factor    = backoff_factor,
            max_parallel_gets=  max_paralell_gets,
            force             = force,
            timeout           = timeout, 
            chunk_size        = chunk_size,
            headers           = headers,
            http2             = http2,
            chunk_index       = chunk_index
        )
        xs:List[Tuple[int, List[PyCtxt]]] = []
        async for (m,data) in get_chunks_generator:
            data_bytes = data.tobytes()
            index = int(m.tags.get("index","-1"))
            h = H.sha256()
            h.update(data_bytes)
            x = pickle.loads(data_bytes)
            xx = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            xs.append((index, xx))
        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
        ordered_xs:List[PyCtxt] = []
        for i in xs_sorted:
            ordered_xs.extend(i[1])
        return ordered_xs


    @staticmethod
    async def get_pyctxt_matrix(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        ckks:Ckks,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] = {},
        http2:bool = False, 
        chunk_index:int =0
    ):
        res = client.get_chunks(
            bucket_id         = bucket_id,
            key               = key,
            timeout           = timeout,
            backoff_factor    = backoff_factor,
            chunk_size        = chunk_size,
            delay             = delay,
            force             = force,
            headers           = headers,
            max_retries       = max_retries,
            http2             = http2,
            max_parallel_gets = max_paralell_gets,
            chunk_index       = chunk_index
        )
        xs = []
        async for (m,c) in res:
            x = Common.from_bytes_to_pyctxt_matrix(ckks= ckks, x = c)
            xs.append(x)
        res = np.vstack(xs)
        return res
    def serialize_matrix_with_pickle(enc_matrix):
        return pickle.dumps(enc_matrix, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize_matrix_with_pickle(serialized_bytes):
        return pickle.loads(serialized_bytes)
    
    @staticmethod
    def segment_and_encrypt_paillier_with_executor(executor:ProcessPoolExecutor,key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, num_chunks:int=2 ):
        plaintext_matrix_chunks:Chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_paillier,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk)
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )
    
    @staticmethod
    def encrypt_chunk_paillier(key:str,dataowner:DataOwnerPHE,chunk:Chunk)-> Chunk:
        ptm = chunk.to_ndarray().unwrap()
        encyrpted_chunk = Common.serialize_matrix_with_pickle(dataowner.paillier_encrypt_matrix_chunk(plaintext_matrix = ptm))
        # return Chunk.from_bytes()
        # print("HERE!", encyrpted_chunk.shape,encyrpted_chunk.dtype)
        return Chunk.from_bytes(group_id=key, index= chunk.index, data= encyrpted_chunk, chunk_id=Some("{}_{}".format(key,chunk.index)),metadata={
            "shape":str(ptm.shape),
            "dtype":str(ptm.dtype)
        } )