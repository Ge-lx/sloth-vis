from quart import Quart, render_template
import state

app = Quart(__name__,
	static_url_path='/',
	static_folder='http_root')

@app.route('/')
async def index():
	return await app.send_static_file('index.html')

@app.route('/pixels/fill')
async def fill():
	state.enable_static()
	return 'OK'

@app.route('/pixels/clear')
async def clear():
	state.disable_static()
	return 'OK'

@app.route('/pixels/timer/<int:secs>')
async def timer(secs):
	# enable timed shutoff
	return 'OK'

@app.route('/pixels/brightness/<int:value>')
async def set_brightness(value):
	state.set_static_brightness(value)
	return 'OK'

@app.route('/pixels/color/<int:r>/<int:g>/<int:b>/<int:w>')
async def set_color(r, g, b, w):
	state.set_static_color({ 'r': r, 'g': g, 'b': b, 'w': w })
	return 'OK'

@app.route('/cava/start')
async def sloth_enable():
	state.enable_visualization()
	return 'OK'

@app.route('/cava/stop')
async def sloth_disable():
	state.disable_visualization()
	return 'OK'

@app.route('/sloth/config/<string:config_name>')
async def sloth_config(config_name):
	state.set_config(config_name)
	return 'OK'

@app.route('/sloth/set/<string:key>/<value>')
async def sloth_key_value(config):
	return 'OK'

@app.route('/status/pixels')
async def status_pixels():
	return state.get_pixels_state()

@app.route('/status/config')
async def status_sloth():
	return state.get_visualization_state()

if __name__ == '__main__':
	app.run(host='0.0.0.0')