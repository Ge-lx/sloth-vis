import sys
from enum import Enum
from queue import Queue
from threading import Lock


# This PulseAudio backend is using the system PulseAudio API via ctypes.
# Bindings for libpulse.so.0 from https://github.com/Valodim/python-pulseaudio
from ctypes import POINTER, c_ubyte, c_void_p, c_ulong, c_char_p, cast, c_ushort, c_int, c_uint, c_uint32, byref, pointer
from audio_backend.lib_pulseaudio import *

def print_structure(s):
    for field in s._fields_:
        print(field[0], getattr(s, field[0]))

def as_dict(struct):
    return dict((field, getattr(struct, field)) for field, _ in struct._fields_)


class AudioInputMode (Enum):
    default_sink = {'idx': 0, 'requires_sink_info_list': True, 'requires_server_info': True},
    sink_by_name = {'idx': 1, 'requires_sink_info_list': True, 'requires_server_info': False},
    default_source ={'idx': 2, 'requires_sink_info_list': False, 'requires_server_info': True},
    source_by_name = {'idx': 3, 'requires_sink_info_list': False, 'requires_server_info': False}


class AsyncCallbackResponseHandler (object):
    def __init__(self, response_identifies, on_callback_data_received):
        self._lock = Lock()
        self.on_callback_data_received = on_callback_data_received;

        self.responses = dict()
        for idf, is_vector, is_multiple in response_identifies:
            self.responses[idf] = {
                'done': False,
                'is_vector': is_vector,
                'is_multiple': is_multiple,
                'data': [] if is_vector else None
            }

    def _validate_idf(self, idf):
        if not idf in self.responses:
            raise Exception(f'Could not find the requested response identifier {idf}')

    def has_response(self, idf):
        with self._lock:
            self._validate_idf(idf)
            is_done = self.responses[idf]['done']

            return is_done or 'previous_data' in self.responses[idf]

    def get_response(self, idf):
        with self._lock:
            self._validate_idf(idf)

            data = None
            if self.responses[idf]['done']:
                data = self.responses[idf]['data']
            elif 'previous_data' in self.responses[idf]:
                data = self.responses['previous_data']
            else:
                return None

            return data

    def add_data(self, idf, data, vector_eol=False):
        # Should we trigger the callback_data_received
        should_trigger = False

        # Lock on the state
        with self._lock:
            self._validate_idf(idf)
            is_multiple = self.responses[idf]['is_multiple']
            is_vector = self.responses[idf]['is_vector']
            is_done = self.responses[idf]['done']

            if not is_multiple and is_done:
                raise Exception(f'Received data for "{idf}" which is already marked as done.')    
            elif not is_vector:
                if data == None:
                    raise Exception(f'Received nullptr for "{idf}"')
                self.responses[idf]['data'] = data
                self.responses[idf]['done'] = True
                should_trigger = True
            else:
                if is_done:
                    self.responses[idf]['previous_data'] = self.responses[idf]['data'] # hopefully this frees the old copy?
                    self.responses[idf]['data'] = []
                    self.responses[idf]['done'] = False # We are now returning from previous_data

                if not data == None:
                    self.responses[idf]['data'].append(data)
                elif vector_eol:
                    self.responses[idf]['done'] = True
                    should_trigger = True
                else:
                    raise Exception(f'Received nullptr for "{idf}"')

        # Trigger the callback outside of the lock
        if should_trigger:
            self.on_callback_data_received(idf, self.get_response(idf))


# based on `peak-detect` by Menno Smits (https://gitlab.com/menn0/peak-detect)
class PulseAudioMonitorClient (object):
    
    def __init__(self, callback_data, callback_meta, input_config, rate=44100, channels=2, period_size=1024):
        self.rate = rate
        self.period_size = period_size
        self.callback_data = callback_data
        self.callback_meta = callback_meta
        self.channels = channels

        cb_response_identifiers = [
            ('context_state', False, True),
            ('server_info', False, False),
            ('sink_info_list', True, False),
            ('timing_info', False, True),
        ]
        self.cb_handler = AsyncCallbackResponseHandler(cb_response_identifiers, self.on_callback_data_received)

        input_config['name'] = '' if not input_config['name'] else input_config['name']
        self.input_mode = input_config['mode']
        self.input_params = input_config['mode'].value[0]
        self.input_name = bytes(input_config['name'], 'utf8')
        print(f'Using audio input mode "{input_config["mode"].name}" (with input name "{input_config["name"]}")')
        
        # Wrap callback methods in appropriate ctypefunc instances so that the Pulseaudio C API can call them
        self._stream_read_cb = pa_stream_request_cb_t(self.stream_read_cb)
        self._stream_timing_info_cb = pa_stream_notify_cb_t(self.stream_timing_info_cb)
        self._context_notify_cb = pa_context_notify_cb_t(self.context_notify_cb)
        self._server_info_cb = pa_server_info_cb_t(self.server_info_cb)
        self._sink_info_list_cb = pa_sink_info_cb_t(self.sink_info_list_cb)

        # Create the mainloop thread and set our context_notify_cb method to be called when there's updates
        # relating to the connection to Pulseaudio
        _mainloop = pa_threaded_mainloop_new()
        _mainloop_api = pa_threaded_mainloop_get_api(_mainloop)
        self.context = pa_context_new(_mainloop_api, b'sloth realtime audio analyzer')
        pa_context_set_state_callback(self.context, self._context_notify_cb, None)
        pa_context_connect(self.context, None, 0, None)
        pa_threaded_mainloop_start(_mainloop)

    def server_info_cb(self, pa_context_p, pa_server_info_p, _):
        server_info = None if not pa_server_info_p else as_dict(pa_server_info_p.contents)
        self.cb_handler.add_data('server_info', server_info)
    
    def sink_info_list_cb(self, pa_context_p, pa_sink_info_p, eol, __):
        sink_info = None if not pa_sink_info_p else as_dict(pa_sink_info_p.contents)
        self.cb_handler.add_data('sink_info_list', sink_info, eol)

    def context_notify_cb(self, pa_context_p, _):
        state = pa_context_get_state(pa_context_p)
        self.cb_handler.add_data('context_state', state)

    def stream_timing_info_cb(self, pa_stream_p, _):
        ## timing information
        timing_info = dict()
        timing_info['general'] = as_dict(pa_stream_get_timing_info(pa_stream_p).contents)

        ## stream latency
        latency = c_ulong()
        negative = c_int() 
        pa_stream_get_latency(pa_stream_p, byref(latency), byref(negative))
        timing_info['stream_latency'] = (-1 if negative.value else 1) * latency.value

        ## buffer attributes
        buffer_attr = pa_stream_get_buffer_attr(pa_stream_p).contents
        sample_spec = pa_stream_get_sample_spec(pa_stream_p).contents
        # print_structure(sample_spec)
        timing_info['buffer_attributes'] = as_dict(buffer_attr)

        self.cb_handler.add_data('timing_info', timing_info)

    def stream_read_cb(self, stream, length, index_incr):
        data = c_void_p()
        pa_stream_peek(stream, data, c_ulong(length))
        l = int(length / 2)
        # print(index_incr)

        data = cast(data, POINTER(c_ushort * l))
        self.callback_data(l, data.contents, index_incr + l)
        pa_stream_drop(stream)

    def on_callback_data_received(self, idf, data):
        if idf == 'timing_info':
            return self.callback_meta(data)

        if idf == 'context_state':
            state = data
            if state == PA_CONTEXT_READY:
                print('Pulseaudio connection ready...')
                # Connected to Pulseaudio. Now request that sink_info_cb
                # be called with information about the available sinks.
                if self.input_params['requires_server_info']:
                    print('Querying pulseaudio server info...')
                    o = pa_context_get_server_info(self.context, self._server_info_cb, None)
                    pa_operation_unref(o)

                if self.input_params['requires_sink_info_list']:
                    print('Querying pulseaudio sink info list...')
                    o = pa_context_get_sink_info_list(self.context, self._sink_info_list_cb, None)
                    pa_operation_unref(o)

            elif state == PA_CONTEXT_FAILED:
                print('Connection failed')

            elif state == PA_CONTEXT_TERMINATED:
                print('Connection terminated')

        if idf in ['server_info', 'sink_info_list']:
            server_info_done = (not self.input_params['requires_server_info']) or self.cb_handler.has_response('server_info')
            sink_info_list_done = (not self.input_params['requires_sink_info_list']) or self.cb_handler.has_response('sink_info_list')

            if server_info_done and sink_info_list_done:
                print('All meta information received sucessfully!')

                source_name = self.input_name
                requires_source_resolution = self.input_mode in [AudioInputMode.default_sink, AudioInputMode.sink_by_name]

                if requires_source_resolution:
                    sink_name = self.input_name

                    if self.input_mode == AudioInputMode.default_sink:
                        sink_name = self.cb_handler.get_response('server_info')['default_sink_name']

                    sink_info_list = self.cb_handler.get_response('sink_info_list')
                    monitor_source_names = [sink['monitor_source_name'] for sink in sink_info_list if sink['name'] == sink_name]
                    source_name = monitor_source_names[0]
                
                elif self.input_mode == AudioInputMode.default_source:
                    source_name = self.cb_handler.get_response('server_info')['default_source_name']

                print(f'Connecting stream to {source_name}')
                self.connect_stream_to_source_by_name(self.context, source_name)

    def connect_stream_to_source_by_name(self, context, name):
        samplespec = pa_sample_spec()
        samplespec.channels = self.channels
        samplespec.format = PA_SAMPLE_S16LE
        samplespec.rate = self.rate

        bytes_per_fragment = pa_frame_size(byref(samplespec)) * self.period_size # = bytes_per_sample * channels
        print(f'Using fragsize of {bytes_per_fragment}')
        
        bufferattr = pa_buffer_attr()
        bufferattr.maxlength = c_uint32(-1) # c_uint32(bytes_per_frame * self.period_size)
        bufferattr.tlength = c_uint32(-1)
        bufferattr.prebuf = c_uint32(-1)
        bufferattr.minreq = c_uint32(-1)
        bufferattr.fragsize = c_uint32(bytes_per_fragment)

        print_structure(samplespec)
        pa_stream = pa_stream_new(context, b"peak detect demo", byref(samplespec), None)
        print_structure(samplespec)
        pa_stream_set_read_callback(pa_stream, self._stream_read_cb, byref(pa_stream))
        pa_stream_set_latency_update_callback(pa_stream, self._stream_timing_info_cb, byref(pa_stream))

        stream_flags = PA_STREAM_AUTO_TIMING_UPDATE
        ret = pa_stream_connect_record(pa_stream, name, byref(bufferattr), stream_flags)

        if not ret == 0:
            raise Exception('Could not connect to monitor source: {}'.format(ret))


client = None
def start_backend(config, callback_data, callback_meta):
    global client

    input_config = {
        'mode': AudioInputMode[config['AUDIO_INPUT_MODE']],
        'name': config['AUDIO_INPUT_NAME']
    }

    client = PulseAudioMonitorClient(callback_data=callback_data,
                                     callback_meta=callback_meta,
                                     input_config=input_config,
                                     rate=config["SAMPLE_RATE"],
                                     channels=config["CHANNELS"],
                                     period_size=config["fft_samples_per_window"])