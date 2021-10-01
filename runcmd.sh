set -- $(hostname -I)
python app.py "$1"
