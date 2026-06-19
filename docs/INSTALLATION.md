## Installation

It is recommended to create a virtual envelope to isolate these pacakges from other environments and projects. 
```bash
virtualenv venv

source venv/bin/activate
```

and then installing the packages. 

```bash
pip install pywebview librosa scikit-learn numpy scipy soundfile
```

On Linux, PyWebView also needs a GTK or Qt backend:

```bash
# either
pip install pywebview[qt]
# or use system GTK web views (Debian/Ubuntu)
sudo apt install python3-gi gir1.2-webkit2-4.1
```
