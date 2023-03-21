venv: venv/touchfile

venv/touchfile: requirements.txt
	pip install virtualenv
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

build: venv
	. venv/bin/activate; echo "Output of pip freeze: " && pip freeze

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

