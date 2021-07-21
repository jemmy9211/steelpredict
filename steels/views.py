from django.shortcuts import render
from PIL import Image
import numpy as np
import io, os, cv2, zlib, threading, argparse
#from module import Modules
# Create your views here.
from django.http import HttpResponse, response

def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

def index(request):
    if request.method=='POST':
        r = request.POST
        """data = uncompress_nparr(r.data)
        color = data[:,:,:3]
        depth = data[:,:,3:6]
        coord = data[:,:,6:9]
        mask, parameter = modules.predict(color, depth, coord)
        info = np.concatenate((mask, parameter), -1)	
        bytestream = io.BytesIO()
        np.save(bytestream, info)
        compressed = zlib.compress(bytestream.getvalue())"""
        return HttpResponse("have request")
    else:
        return HttpResponse("no request!")
    

    