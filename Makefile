.PHONY: build setup-models hardware config llm-test test shell

IMAGE_NAME := localhost/graphy:latest
EXTRA_PKG_CACHE := /tmp/opencode/pacman_extra
AI_RUNTIME_CACHE := /tmp/opencode/ai_runtime

build:
	mkdir -p $(EXTRA_PKG_CACHE) $(AI_RUNTIME_CACHE)
	podman build \
		-v /var/cache/pacman/pkg:/var/cache/pacman/pkg:ro \
		-v $(EXTRA_PKG_CACHE):/cache_extra:ro \
		-v $(AI_RUNTIME_CACHE):/ai_runtime:ro \
		-v /var/lib/pacman/sync:/var/lib/pacman/sync:ro \
		-v /etc/pacman.d/mirrorlist:/etc/pacman.d/mirrorlist:ro \
		-t $(IMAGE_NAME) -f Containerfile .

setup-models:
	python3 scripts/setup_models.py

hardware:
	./scripts/podman_run.sh hardware

config:
	./scripts/podman_run.sh config

llm-test:
	./scripts/podman_run.sh llm-test

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
		-v "$(CURDIR)/scripts:/app/scripts:ro" \
		-v "$(CURDIR)/docs:/app/docs:ro" \
		-v "$(CURDIR)/README.md:/app/README.md:ro" \
		-v "$(CURDIR)/CONTAINER_DETAILS.md:/app/CONTAINER_DETAILS.md:ro" \
		-v "$(CURDIR)/Containerfile:/app/Containerfile:ro" \
		-v "$(CURDIR)/compose.yaml:/app/compose.yaml:ro" \
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
