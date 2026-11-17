# usage: python vid_dots.py clip.mjpeg ...  (laser dot per frame in a 300 px window around the bore estimate)
import sys, cv2, numpy as np, json
def frames(path):
    data=open(path,'rb').read(); i=0
    while True:
        s=data.find(b'\xff\xd8',i)
        if s<0: return
        e=data.find(b'\xff\xd9',s)
        if e<0: return
        yield cv2.imdecode(np.frombuffer(data[s:e+2],np.uint8),cv2.IMREAD_GRAYSCALE); i=e+2
X0,Y0,R=524,508,150
out=[]
for path in sys.argv[1:]:
    for k,im in enumerate(frames(path)):
        if im is None: continue
        roi=im[Y0-R:Y0+R,X0-R:X0+R].astype(np.float32)
        dog=cv2.GaussianBlur(roi,(0,0),1.5)-cv2.GaussianBlur(roi,(0,0),6)
        noise=np.median(np.abs(dog))*1.48+1e-3
        y,x=np.unravel_index(np.argmax(dog),dog.shape); snr=dog[y,x]/noise
        # second peak away from first
        d2=dog.copy(); d2[max(0,y-12):y+12,max(0,x-12):x+12]=-1e9; snr2=d2.max()/noise
        ya,yb,xa,xb=max(0,y-7),min(dog.shape[0],y+8),max(0,x-7),min(dog.shape[1],x+8); w=dog[ya:yb,xa:xb].clip(0); yy,xx=np.mgrid[ya:yb,xa:xb]
        cx=(w*xx).sum()/w.sum(); cy=(w*yy).sum()/w.sum()
        m=cv2.moments(w); mu20,mu02,mu11=m['mu20']/m['m00'],m['mu02']/m['m00'],m['mu11']/m['m00']
        ev=np.linalg.eigvalsh([[mu20,mu11],[mu11,mu02]]); elong=float(np.sqrt(ev[1]/max(ev[0],1e-6)))
        out.append(dict(clip=path.split('/')[-1],i=k,x=cx+X0-R,y=cy+Y0-R,snr=float(snr),snr2=float(snr2),elong=elong,peak=float(roi[y,x])))
json.dump(out,open('vid_dots.json','w'))
print(len(out),"frames")
