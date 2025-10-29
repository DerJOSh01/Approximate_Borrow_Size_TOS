EXTERNAL_DIR := /tmp/vendor/Approximate_Borrow_Size_TOS$(shell bash -c 'echo $$RANDOM')
EXTERNAL_REPO := https://github.com/DerJOSh01/Approximate_Borrow_Size_TOS
EXTERNAL_MK := $(EXTERNAL_DIR)/Makefile

.PHONY: all clean download_conda webtop-up webtop-down app-up sync-code

download_conda:
	curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

webtop-up:
	docker compose up -d
	docker compose logs -f

webtop-down:
	docker compose down

app-up:
	docker build -t docker-xterm -f Dockerfile.runtime . 
	docker run --rm -p 5800:5800 -p 5900:5900 -v .:/codespace docker-xterm

# Use this command to update if you make your git-repo private
sync-code:
	if [ ! -f "$(EXTERNAL_MK)" ]; then \
		echo "Cloning external dependency..."; \
		mkdir -p $(dir $(EXTERNAL_MK)); \
		git clone --depth 1 --branch main $(EXTERNAL_REPO) $(EXTERNAL_DIR); \
	fi; \

	rsync -av --exclude='.*' "$(EXTERNAL_DIR)/" "./"