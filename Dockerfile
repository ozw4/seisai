# syntax=docker/dockerfile:1

# uv/uvx をコピーするためのステージ（distroless: uv バイナリのみ）
FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

# uv/uvx を本体イメージへ
COPY --from=uvbin /uv /uvx /usr/local/bin/

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    git \
    git-lfs \
    shellcheck \
    fontconfig \
    ttf-mscorefonts-installer \
    && git lfs install --system \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && mkdir -p -m 755 /etc/apt/sources.list.d \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs gh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

# Codex CLI
RUN npm i -g @openai/codex

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=requirements-dev.txt \
    python -m pip install -r requirements-dev.txt

RUN addgroup --gid $GID $USERNAME && \
    adduser --disabled-password --gecos "" --shell "/bin/bash" --uid $UID --gid $GID $USERNAME

USER $USERNAME

COPY ruff.toml /home/$USERNAME
ENV PYTHONPATH="/workspace:${PYTHONPATH}:"
ENV PATH="$HOME/.local/bin:$PATH"
WORKDIR /workspace

CMD ["/bin/bash"]
