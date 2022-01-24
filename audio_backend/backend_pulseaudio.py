import sys
from queue import Queue
from ctypes import POINTER, c_ubyte, c_void_p, c_ulong, c_char_p, cast, c_ushort, c_int, c_uint, c_uint32, byref

# From https://github.com/Valodim/python-pulseaudio
from audio_backend.lib_pulseaudio import *
def print_structure (s):
    for field in s._fields_:
        print(field[0], getattr(s, field[0]))

# based on `peak-detect` by Menno Smits (https://gitlab.com/menn0/peak-detect)
class PulseAudioMonitorClient (object):
    
    def __init__(self, sink_name, rate, channels, period_size, callback):
        self.sink_name = sink_name
        self.rate = rate
        self.period_size = period_size
        self.callback = callback
        self.channels = channels

        # Wrap callback methods in appropriate ctypefunc instances so
        # that the Pulseaudio C API can call them
        self._context_notify_cb = pa_context_notify_cb_t(self.context_notify_cb)
        self._sink_info_cb = pa_sink_info_cb_t(self.sink_info_cb)
        self._stream_read_cb = pa_stream_request_cb_t(self.stream_read_cb)

        # Create the mainloop thread and set our context_notify_cb
        # method to be called when there's updates relating to the
        # connection to Pulseaudio
        _mainloop = pa_threaded_mainloop_new()
        _mainloop_api = pa_threaded_mainloop_get_api(_mainloop)
        context = pa_context_new(_mainloop_api, b'peak demo')
        pa_context_set_state_callback(context, self._context_notify_cb, None)
        pa_context_connect(context, None, 0, None)
        pa_threaded_mainloop_start(_mainloop)


    def context_notify_cb(self, context, _):
        state = pa_context_get_state(context)

        if state == PA_CONTEXT_READY:
            print( "Pulseaudio connection ready...")
            # Connected to Pulseaudio. Now request that sink_info_cb
            # be called with information about the available sinks.
            o = pa_context_get_sink_info_list(context, self._sink_info_cb, None)
            pa_operation_unref(o)

        elif state == PA_CONTEXT_FAILED :
            print( "Connection failed")

        elif state == PA_CONTEXT_TERMINATED:
            print( "Connection terminated")

    def sink_info_cb(self, context, sink_info_p, _, __):
        if not sink_info_p:
            return

        try:
            sink_info = sink_info_p.contents
            print( '-'* 60)
            print( 'index:', sink_info.index)
            print( 'name:', sink_info.name)
            print( 'description:', sink_info.description)

            if sink_info.name == bytes(self.sink_name, 'utf8'):
                # Found the sink we want to monitor for peak levels.
                # Tell PA to call stream_read_cb with peak samples.
                print()
                print('setting up peak recording using', sink_info.monitor_source_name)
                print()
                print('sink_info.sample_spec:')
                print_structure(sink_info.sample_spec)
                print()

                samplespec = pa_sample_spec()
                samplespec.channels = self.channels
                samplespec.format = PA_SAMPLE_S16LE
                samplespec.rate = self.rate

                bytes_per_frame = pa_frame_size(byref(samplespec)) # = bytes_per_sample * channels
                buffer_length_usec = pa_bytes_to_usec(self.period_size, byref(samplespec))
                print(f'buffer_length_usec = {buffer_length_usec}')

                bufferattr = pa_buffer_attr()
                bufferattr.maxlength = c_uint32(-1)#c_uint32(bytes_per_frame * self.period_size)
                bufferattr.tlength = c_uint32(-1)
                bufferattr.prebuf = c_uint32(-1)
                bufferattr.minreq = c_uint32(-1)
                bufferattr.fragsize = c_uint32(bytes_per_frame * self.period_size)
                print(f'Using fragsize of {bufferattr.fragsize}')

                pa_stream = pa_stream_new(context, b"peak detect demo", byref(samplespec), None)
                pa_stream_set_read_callback(pa_stream,
                                            self._stream_read_cb,
                                            byref(pa_stream))
                ret = pa_stream_connect_record(pa_stream,
                                         sink_info.monitor_source_name,
                                         byref(bufferattr),
                                         PA_STREAM_ADJUST_LATENCY | PA_STREAM_AUTO_TIMING_UPDATE)
                if ret != 0:
                    raise Error('Could not connect to monitor source: {}'.format(ret))
        except Exception as e:
            print('SOMETHIGN WEHTN WRONG: ', e)

        
    def stream_read_cb(self, stream, length, index_incr):
        data = c_void_p()
        pa_stream_peek(stream, data, c_ulong(length))

        l = int(length / 2)

        data = cast(data, POINTER(c_ushort * l))
        self.callback(l, data.contents)
    
        ### DEBUG PulseAudio Stream Properties

        ### latency
        # latency = c_ulong()
        # negative = c_int() 
        # pa_stream_get_latency(stream, byref(latency), byref(negative))
        # print(f'latency: {latency},\nnegative: {negative}')

        ### buffer attributes
        # buffer_attr = pa_stream_get_buffer_attr(stream).contents
        # print_structure(buffer_attr)

        ### timing information
        # timing_info = pa_stream_get_timing_info(stream).contents
        # print_structure(timing_info)

        # print('\n\n')

        pa_stream_drop(stream)

client = None

def start_backend(config, callback):
    global client
    client = PulseAudioMonitorClient(sink_name=config["PULSEAUDIO_SINK"],
                                     rate=config["SAMPLE_RATE"],
                                     channels=config["CHANNELS"],
                                     period_size=config["fft_samples_per_window"],
                                     callback=callback);
    print('Backend started.')