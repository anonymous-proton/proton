"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'modelworker.proto'
)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11modelworker.proto\x12\x0emodelworker.v1\"\x07\n\x05\x45mpty\"\"\n\rHealthRequest\x12\x11\n\treadiness\x18\x01 \x01(\x08\"-\n\x0eHealthResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"\xfd\x01\n\x0c\x43\x61pabilities\x12\x11\n\tworker_id\x18\x01 \x01(\t\x12\x12\n\nmodel_name\x18\x02 \x01(\t\x12\x15\n\rmodel_version\x18\x03 \x01(\t\x12\x19\n\x11supported_buckets\x18\n \x03(\t\x12\x16\n\x0emax_batch_size\x18\x0b \x01(\x05\x12\x1c\n\x14max_inflight_batches\x18\x0c \x01(\x05\x12\x1e\n\x16supports_prepare_batch\x18\x14 \x01(\x08\x12\x1f\n\x17supports_finalize_batch\x18\x15 \x01(\x08\x12\x1d\n\x15supports_cancel_batch\x18\x16 \x01(\x08\"\x8d\x02\n\nQueueStats\x12\x10\n\x08in_queue\x18\x01 \x01(\x05\x12\x16\n\x0eprepared_queue\x18\x02 \x01(\x05\x12\x14\n\x0coutput_queue\x18\x03 \x01(\x05\x12\x18\n\x10prepare_inflight\x18\x04 \x01(\x05\x12\x18\n\x10\x65xecute_inflight\x18\x05 \x01(\x05\x12\x19\n\x11\x66inalize_inflight\x18\x06 \x01(\x05\x12\x1b\n\x13prepare_concurrency\x18\x07 \x01(\x05\x12\x1b\n\x13\x65xecute_concurrency\x18\x08 \x01(\x05\x12\x1c\n\x14\x66inalize_concurrency\x18\t \x01(\x05\x12\x18\n\x10item_concurrency\x18\n \x01(\x05\"\xb8\x01\n\x0bWorkerStats\x12\x11\n\tworker_id\x18\x01 \x01(\t\x12*\n\x06queues\x18\x02 \x01(\x0b\x32\x1a.modelworker.v1.QueueStats\x12\x43\n\x11resident_baseline\x18\x03 \x01(\x0b\x32(.modelworker.v1.ResidentBaselineSnapshot\x12%\n\x1d\x62ootstrap_memory_summary_json\x18\x04 \x01(\x0c\"\xcb\x01\n\x18ResidentBaselineSnapshot\x12\x1b\n\x13resident_memory_mib\x18\x01 \x01(\x01\x12\x1e\n\x16resident_memory_source\x18\x02 \x01(\t\x12&\n\x1eresident_baseline_collected_at\x18\x03 \x01(\x01\x12)\n!resident_baseline_lifecycle_token\x18\x04 \x01(\t\x12\x1f\n\x17resident_baseline_state\x18\x05 \x01(\t\"\x9f\x01\n\x0bRequestItem\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x14\n\x0cpayload_json\x18\x02 \x01(\x0c\x12\x37\n\x06params\x18\x03 \x03(\x0b\x32\'.modelworker.v1.RequestItem.ParamsEntry\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"t\n\x0b\x42\x61tchTiming\x12\x16\n\x0equeue_delay_us\x18\x01 \x01(\x03\x12\x12\n\nprepare_us\x18\x02 \x01(\x03\x12\x12\n\nexecute_us\x18\x03 \x01(\x03\x12\x13\n\x0b\x66inalize_us\x18\x04 \x01(\x03\x12\x10\n\x08total_us\x18\x05 \x01(\x03\"[\n\x0cResponseItem\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\n\n\x02ok\x18\x02 \x01(\x08\x12\x14\n\x0cpayload_json\x18\x03 \x01(\x0c\x12\x15\n\rerror_message\x18\x04 \x01(\t\"\xcb\x01\n\x0c\x42\x61tchRequest\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x01(\x04\x12\x11\n\tbucket_id\x18\x02 \x01(\t\x12\x38\n\x06params\x18\x03 \x03(\x0b\x32(.modelworker.v1.BatchRequest.ParamsEntry\x12-\n\x08requests\x18\x04 \x03(\x0b\x32\x1b.modelworker.v1.RequestItem\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xa8\x02\n\rBatchResponse\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x01(\x04\x12/\n\tresponses\x18\x02 \x03(\x0b\x32\x1c.modelworker.v1.ResponseItem\x12+\n\x06timing\x18\x03 \x01(\x0b\x32\x1b.modelworker.v1.BatchTiming\x12\x38\n\x0b\x61ttribution\x18\x04 \x01(\x0b\x32#.modelworker.v1.DispatchAttribution\x12%\n\x1d\x62ootstrap_memory_summary_json\x18\x05 \x01(\x0c\x12#\n\x1b\x64ispatch_memory_window_json\x18\x06 \x01(\x0c\x12\n\n\x02ok\x18\n \x01(\x08\x12\x15\n\rerror_message\x18\x0b \x01(\t\"t\n\x13\x44ispatchAttribution\x12\x1f\n\x17worker_generation_token\x18\x01 \x01(\t\x12!\n\x19run_ordinal_in_generation\x18\x02 \x01(\x05\x12\x19\n\x11is_first_real_run\x18\x03 \x01(\x08\"\xf0\x01\n\x11SubmitTaskRequest\x12\x12\n\nnf_task_id\x18\x01 \x01(\t\x12\x11\n\tcomponent\x18\x02 \x01(\t\x12\x0f\n\x07workdir\x18\x03 \x01(\t\x12\x16\n\x0e\x63ommand_script\x18\x04 \x01(\t\x12\x13\n\x0bmain_script\x18\x05 \x01(\t\x12\x37\n\x03\x65nv\x18\x06 \x03(\x0b\x32*.modelworker.v1.SubmitTaskRequest.EnvEntry\x12\x11\n\ttimeout_s\x18\x07 \x01(\r\x1a*\n\x08\x45nvEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"%\n\x12SubmitTaskResponse\x12\x0f\n\x07task_id\x18\x01 \x01(\t\"!\n\x0eGetTaskRequest\x12\x0f\n\x07task_id\x18\x01 \x01(\t\"\x93\x01\n\x0fGetTaskResponse\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12(\n\x05state\x18\x02 \x01(\x0e\x32\x19.modelworker.v1.TaskState\x12\n\n\x02ok\x18\n \x01(\x08\x12\x11\n\texit_code\x18\x0b \x01(\x05\x12\x0f\n\x07message\x18\x0c \x01(\t\x12\x15\n\rmanifest_json\x18\r \x01(\x0c\"$\n\x11\x43\x61ncelTaskRequest\x12\x0f\n\x07task_id\x18\x01 \x01(\t\"1\n\x12\x43\x61ncelTaskResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"&\n\x12\x43\x61ncelBatchRequest\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x01(\x05\"2\n\x13\x43\x61ncelBatchResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t*\xa4\x01\n\tTaskState\x12\x1a\n\x16TASK_STATE_UNSPECIFIED\x10\x00\x12\x18\n\x14TASK_STATE_SUBMITTED\x10\x01\x12\x16\n\x12TASK_STATE_RUNNING\x10\x02\x12\x18\n\x14TASK_STATE_SUCCEEDED\x10\x03\x12\x15\n\x11TASK_STATE_FAILED\x10\x04\x12\x18\n\x14TASK_STATE_CANCELLED\x10\x05\x32\x81\x03\n\x0bModelWorker\x12\x46\n\x0fGetCapabilities\x12\x15.modelworker.v1.Empty\x1a\x1c.modelworker.v1.Capabilities\x12I\n\nInferBatch\x12\x1c.modelworker.v1.BatchRequest\x1a\x1d.modelworker.v1.BatchResponse\x12G\n\x06Health\x12\x1d.modelworker.v1.HealthRequest\x1a\x1e.modelworker.v1.HealthResponse\x12>\n\x08GetStats\x12\x15.modelworker.v1.Empty\x1a\x1b.modelworker.v1.WorkerStats\x12V\n\x0b\x43\x61ncelBatch\x12\".modelworker.v1.CancelBatchRequest\x1a#.modelworker.v1.CancelBatchResponse2\xff\x01\n\x07Gateway\x12S\n\nSubmitTask\x12!.modelworker.v1.SubmitTaskRequest\x1a\".modelworker.v1.SubmitTaskResponse\x12J\n\x07GetTask\x12\x1e.modelworker.v1.GetTaskRequest\x1a\x1f.modelworker.v1.GetTaskResponse\x12S\n\nCancelTask\x12!.modelworker.v1.CancelTaskRequest\x1a\".modelworker.v1.CancelTaskResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'modelworker_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_REQUESTITEM_PARAMSENTRY']._loaded_options = None
  _globals['_REQUESTITEM_PARAMSENTRY']._serialized_options = b'8\001'
  _globals['_BATCHREQUEST_PARAMSENTRY']._loaded_options = None
  _globals['_BATCHREQUEST_PARAMSENTRY']._serialized_options = b'8\001'
  _globals['_SUBMITTASKREQUEST_ENVENTRY']._loaded_options = None
  _globals['_SUBMITTASKREQUEST_ENVENTRY']._serialized_options = b'8\001'
  _globals['_TASKSTATE']._serialized_start=2695
  _globals['_TASKSTATE']._serialized_end=2859
  _globals['_EMPTY']._serialized_start=37
  _globals['_EMPTY']._serialized_end=44
  _globals['_HEALTHREQUEST']._serialized_start=46
  _globals['_HEALTHREQUEST']._serialized_end=80
  _globals['_HEALTHRESPONSE']._serialized_start=82
  _globals['_HEALTHRESPONSE']._serialized_end=127
  _globals['_CAPABILITIES']._serialized_start=130
  _globals['_CAPABILITIES']._serialized_end=383
  _globals['_QUEUESTATS']._serialized_start=386
  _globals['_QUEUESTATS']._serialized_end=655
  _globals['_WORKERSTATS']._serialized_start=658
  _globals['_WORKERSTATS']._serialized_end=842
  _globals['_RESIDENTBASELINESNAPSHOT']._serialized_start=845
  _globals['_RESIDENTBASELINESNAPSHOT']._serialized_end=1048
  _globals['_REQUESTITEM']._serialized_start=1051
  _globals['_REQUESTITEM']._serialized_end=1210
  _globals['_REQUESTITEM_PARAMSENTRY']._serialized_start=1165
  _globals['_REQUESTITEM_PARAMSENTRY']._serialized_end=1210
  _globals['_BATCHTIMING']._serialized_start=1212
  _globals['_BATCHTIMING']._serialized_end=1328
  _globals['_RESPONSEITEM']._serialized_start=1330
  _globals['_RESPONSEITEM']._serialized_end=1421
  _globals['_BATCHREQUEST']._serialized_start=1424
  _globals['_BATCHREQUEST']._serialized_end=1627
  _globals['_BATCHREQUEST_PARAMSENTRY']._serialized_start=1165
  _globals['_BATCHREQUEST_PARAMSENTRY']._serialized_end=1210
  _globals['_BATCHRESPONSE']._serialized_start=1630
  _globals['_BATCHRESPONSE']._serialized_end=1926
  _globals['_DISPATCHATTRIBUTION']._serialized_start=1928
  _globals['_DISPATCHATTRIBUTION']._serialized_end=2044
  _globals['_SUBMITTASKREQUEST']._serialized_start=2047
  _globals['_SUBMITTASKREQUEST']._serialized_end=2287
  _globals['_SUBMITTASKREQUEST_ENVENTRY']._serialized_start=2245
  _globals['_SUBMITTASKREQUEST_ENVENTRY']._serialized_end=2287
  _globals['_SUBMITTASKRESPONSE']._serialized_start=2289
  _globals['_SUBMITTASKRESPONSE']._serialized_end=2326
  _globals['_GETTASKREQUEST']._serialized_start=2328
  _globals['_GETTASKREQUEST']._serialized_end=2361
  _globals['_GETTASKRESPONSE']._serialized_start=2364
  _globals['_GETTASKRESPONSE']._serialized_end=2511
  _globals['_CANCELTASKREQUEST']._serialized_start=2513
  _globals['_CANCELTASKREQUEST']._serialized_end=2549
  _globals['_CANCELTASKRESPONSE']._serialized_start=2551
  _globals['_CANCELTASKRESPONSE']._serialized_end=2600
  _globals['_CANCELBATCHREQUEST']._serialized_start=2602
  _globals['_CANCELBATCHREQUEST']._serialized_end=2640
  _globals['_CANCELBATCHRESPONSE']._serialized_start=2642
  _globals['_CANCELBATCHRESPONSE']._serialized_end=2692
  _globals['_MODELWORKER']._serialized_start=2862
  _globals['_MODELWORKER']._serialized_end=3247
  _globals['_GATEWAY']._serialized_start=3250
  _globals['_GATEWAY']._serialized_end=3505
