# codehalter bootstrap assisted
FROM quay.io/archlinux/archlinux:base-devel

RUN pacman -Syu --noconfirm --needed \
        ca-certificates \
        curl \
        git \
        openssh \
        firefox \
    && rm -rf /var/cache/pacman/pkg/*

RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash -G wheel dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && install -d -o 1000 -g 1000 /home/dev/.config

USER dev
WORKDIR /home/dev
# AUR helper for 3rd-choice installs (see SKILL-arch.md). yay-bin is the
# prebuilt AUR package, so this is one makepkg run, not a Go compile; the
# base-devel image ships makepkg and the dev user's NOPASSWD sudo covers the
# dependency installs.
RUN git clone https://aur.archlinux.org/yay-bin.git /tmp/yay-bin \
    && cd /tmp/yay-bin && makepkg -si --noconfirm \
    && rm -rf /tmp/yay-bin
# install.sh installs to ~/.local/bin as the non-root dev user; put it on PATH.
ENV PATH="/home/dev/.local/bin:${PATH}"
# ADD re-fetches the tags JSON every build (via ETag): a new codehalter release
# changes it and busts the cache below; otherwise the install layer stays cached.
ADD https://api.github.com/repos/tbocek/codehalter/tags /tmp/.codehalter-tags.json
RUN curl -sL https://raw.githubusercontent.com/tbocek/codehalter/main/install.sh | bash
