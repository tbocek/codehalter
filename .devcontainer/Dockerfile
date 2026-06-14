# codehalter bootstrap assisted
FROM quay.io/archlinux/archlinux:base-devel

RUN pacman -Syu --noconfirm --needed \
        ca-certificates \
        curl \
        git \
        openssh \
        go \
        gopls \
        just \
        firefox \
    && rm -rf /var/cache/pacman/pkg/*

RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash -G wheel dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && install -d -o 1000 -g 1000 /home/dev/.config

RUN curl -sL https://raw.githubusercontent.com/tbocek/codehalter/main/install.sh | bash

USER dev
WORKDIR /home/dev

RUN git clone https://aur.archlinux.org/yay.git /tmp/yay \
    && cd /tmp/yay && makepkg -si --noconfirm \
    && cd / && rm -rf /tmp/yay
