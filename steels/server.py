from flask import Flask, request, Response, json
import numpy as np
import io, os, cv2, zlib, threading, argparse
from PIL import Image
from module import Modules

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip', type=str, help='IP on server', default="192.168.50.80")
args = parser.parse_args()

SERVER_HOST = args.ip
SERVER_PORT = 5000
API_PATH = '/upload'

def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

app = Flask(__name__)

@app.route(API_PATH, methods=['POST'])
def test():
	r = request
	data = uncompress_nparr(r.data)
	color = data[:,:,:3]
	depth = data[:,:,3:6]
	coord = data[:,:,6:9]
	mask, parameter = modules.predict(color, depth, coord)
	info = np.concatenate((mask, parameter), -1)	
	bytestream = io.BytesIO()
	np.save(bytestream, info)
	compressed = zlib.compress(bytestream.getvalue())
	return Response(response=compressed, status=200, content_type="application/octet_stream")
	
	
	
if __name__ == "__main__":
	modules = Modules(device='cuda')
	app.run(host=SERVER_HOST, port=SERVER_PORT)
