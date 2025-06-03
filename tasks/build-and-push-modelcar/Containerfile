FROM registry.access.redhat.com/ubi8/python-312

USER root
# Install system dependencies
RUN dnf update -y && dnf install -y \
    git \
    make \
    golang \
    skopeo \
    curl \
    gcc \
    gcc-c++ \
    pkgconfig \
    openssl-devel \
    && dnf clean all

# Install Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install setuptools-rust

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.4.2

# Create Poetry cache directory and set permissions
RUN mkdir -p /opt/app-root/src/.cache/pypoetry && \
    chown -R 1001:0 /opt/app-root/src/.cache && \
    chmod -R g+rwx /opt/app-root/src/.cache

# Add Poetry to PATH
ENV PATH="/opt/app-root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER 1001 