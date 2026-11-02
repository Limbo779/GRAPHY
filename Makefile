.PHONY: build hardware config test shell

IMAGE_NAME := localhost/graphy:latest
EXTRA_PKG_CACHE := /tmp/opencode/pacman_extra

build:
	mkdir -p $(EXTRA_PKG_CACHE)
	podman build \
		-v /var/cache/pacman/pkg:/var/cache/pacman/pkg:ro \
		-v $(EXTRA_PKG_CACHE):/cache_extra:ro \
		-v /var/lib/pacman/sync:/var/lib/pacman/sync:ro \
		-v /etc/pacman.d/mirrorlist:/etc/pacman.d/mirrorlist:ro \
		-t $(IMAGE_NAME) -f Containerfile .

hardware:
	./scripts/podman_run.sh hardware

config:
	./scripts/podman_run.sh config

test:
	podman run --rm \
		--device /dev/dri:/dev/dri:rw \
		--group-add keep-groups \
		--security-opt label=disable \
		--memory 8g \
		-v "$(CURDIR)/models:/app/models:rw" \
		-v "$(CURDIR)/data:/app/data:rw" \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/src:/app/src:ro" \
		-v "$(CURDIR)/tests:/app/tests:ro" \
		--entrypoint python3 \
		$(IMAGE_NAME) -m unittest discover -s /app/tests -v

shell:
	podman run --rm -it \
		--device /dev/dri:/dev/dri:rw \
		--group-add keep-groups \
		--security-opt label=disable \
		--memory 8g \
		-v "$(CURDIR)/models:/app/models:rw" \
		-v "$(CURDIR)/data:/app/data:rw" \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/src:/app/src:ro" \
		-v "$(CURDIR)/tests:/app/tests:ro" \
		--entrypoint bash \
		$(IMAGE_NAME)
