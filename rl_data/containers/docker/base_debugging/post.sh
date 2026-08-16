export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dbg \
    coreutils findutils gawk sed grep \
    curl wget git \
    build-essential gcc g++ make cmake \
    gdb valgrind strace ltrace \
    binutils file \
    jq \
    sudo \
    golang-go \
    rustc cargo \
    tcpdump \
&& rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir pytest pyyaml requests

useradd -m -s /bin/bash user || true
chmod 755 /home/user
