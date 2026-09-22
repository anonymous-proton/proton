"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import modelworker_pb2 as modelworker__pb2

GRPC_GENERATED_VERSION = '1.76.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + ' but the generated code in modelworker_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ModelWorkerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetCapabilities = channel.unary_unary(
                '/modelworker.v1.ModelWorker/GetCapabilities',
                request_serializer=modelworker__pb2.Empty.SerializeToString,
                response_deserializer=modelworker__pb2.Capabilities.FromString,
                _registered_method=True)
        self.InferBatch = channel.unary_unary(
                '/modelworker.v1.ModelWorker/InferBatch',
                request_serializer=modelworker__pb2.BatchRequest.SerializeToString,
                response_deserializer=modelworker__pb2.BatchResponse.FromString,
                _registered_method=True)
        self.Health = channel.unary_unary(
                '/modelworker.v1.ModelWorker/Health',
                request_serializer=modelworker__pb2.HealthRequest.SerializeToString,
                response_deserializer=modelworker__pb2.HealthResponse.FromString,
                _registered_method=True)
        self.GetStats = channel.unary_unary(
                '/modelworker.v1.ModelWorker/GetStats',
                request_serializer=modelworker__pb2.Empty.SerializeToString,
                response_deserializer=modelworker__pb2.WorkerStats.FromString,
                _registered_method=True)
        self.CancelBatch = channel.unary_unary(
                '/modelworker.v1.ModelWorker/CancelBatch',
                request_serializer=modelworker__pb2.CancelBatchRequest.SerializeToString,
                response_deserializer=modelworker__pb2.CancelBatchResponse.FromString,
                _registered_method=True)


class ModelWorkerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetCapabilities(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def InferBatch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Health(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetStats(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CancelBatch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelWorkerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetCapabilities': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCapabilities,
                    request_deserializer=modelworker__pb2.Empty.FromString,
                    response_serializer=modelworker__pb2.Capabilities.SerializeToString,
            ),
            'InferBatch': grpc.unary_unary_rpc_method_handler(
                    servicer.InferBatch,
                    request_deserializer=modelworker__pb2.BatchRequest.FromString,
                    response_serializer=modelworker__pb2.BatchResponse.SerializeToString,
            ),
            'Health': grpc.unary_unary_rpc_method_handler(
                    servicer.Health,
                    request_deserializer=modelworker__pb2.HealthRequest.FromString,
                    response_serializer=modelworker__pb2.HealthResponse.SerializeToString,
            ),
            'GetStats': grpc.unary_unary_rpc_method_handler(
                    servicer.GetStats,
                    request_deserializer=modelworker__pb2.Empty.FromString,
                    response_serializer=modelworker__pb2.WorkerStats.SerializeToString,
            ),
            'CancelBatch': grpc.unary_unary_rpc_method_handler(
                    servicer.CancelBatch,
                    request_deserializer=modelworker__pb2.CancelBatchRequest.FromString,
                    response_serializer=modelworker__pb2.CancelBatchResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'modelworker.v1.ModelWorker', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('modelworker.v1.ModelWorker', rpc_method_handlers)


class ModelWorker(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetCapabilities(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.ModelWorker/GetCapabilities',
            modelworker__pb2.Empty.SerializeToString,
            modelworker__pb2.Capabilities.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def InferBatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.ModelWorker/InferBatch',
            modelworker__pb2.BatchRequest.SerializeToString,
            modelworker__pb2.BatchResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Health(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.ModelWorker/Health',
            modelworker__pb2.HealthRequest.SerializeToString,
            modelworker__pb2.HealthResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetStats(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.ModelWorker/GetStats',
            modelworker__pb2.Empty.SerializeToString,
            modelworker__pb2.WorkerStats.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def CancelBatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.ModelWorker/CancelBatch',
            modelworker__pb2.CancelBatchRequest.SerializeToString,
            modelworker__pb2.CancelBatchResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class GatewayStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SubmitTask = channel.unary_unary(
                '/modelworker.v1.Gateway/SubmitTask',
                request_serializer=modelworker__pb2.SubmitTaskRequest.SerializeToString,
                response_deserializer=modelworker__pb2.SubmitTaskResponse.FromString,
                _registered_method=True)
        self.GetTask = channel.unary_unary(
                '/modelworker.v1.Gateway/GetTask',
                request_serializer=modelworker__pb2.GetTaskRequest.SerializeToString,
                response_deserializer=modelworker__pb2.GetTaskResponse.FromString,
                _registered_method=True)
        self.CancelTask = channel.unary_unary(
                '/modelworker.v1.Gateway/CancelTask',
                request_serializer=modelworker__pb2.CancelTaskRequest.SerializeToString,
                response_deserializer=modelworker__pb2.CancelTaskResponse.FromString,
                _registered_method=True)


class GatewayServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SubmitTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CancelTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GatewayServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SubmitTask': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitTask,
                    request_deserializer=modelworker__pb2.SubmitTaskRequest.FromString,
                    response_serializer=modelworker__pb2.SubmitTaskResponse.SerializeToString,
            ),
            'GetTask': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTask,
                    request_deserializer=modelworker__pb2.GetTaskRequest.FromString,
                    response_serializer=modelworker__pb2.GetTaskResponse.SerializeToString,
            ),
            'CancelTask': grpc.unary_unary_rpc_method_handler(
                    servicer.CancelTask,
                    request_deserializer=modelworker__pb2.CancelTaskRequest.FromString,
                    response_serializer=modelworker__pb2.CancelTaskResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'modelworker.v1.Gateway', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('modelworker.v1.Gateway', rpc_method_handlers)


class Gateway(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SubmitTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.Gateway/SubmitTask',
            modelworker__pb2.SubmitTaskRequest.SerializeToString,
            modelworker__pb2.SubmitTaskResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.Gateway/GetTask',
            modelworker__pb2.GetTaskRequest.SerializeToString,
            modelworker__pb2.GetTaskResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def CancelTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelworker.v1.Gateway/CancelTask',
            modelworker__pb2.CancelTaskRequest.SerializeToString,
            modelworker__pb2.CancelTaskResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
