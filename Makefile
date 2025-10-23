download_conda:
	curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

webtop-up:
	docker compose up -d
	docker compose logs -f

webtop-down:
	docker compose down

app-up:
	cd x-server && docker build -t docker-xterm . && cd ..
	docker run --rm -p 5800:5800 -p 5900:5900 -v .:/codespace docker-xterm
