obj-m += pmu_module.o

# ── platform detection ─────────────────────────────────────
# The Milk-V Pioneer (SG2042 / T-Head C920) needs -DTARGET_MILKV_PIONEER for
# its cache geometry (see pmu_scanner.c). Detected from the device-tree
# compatible strings (board root node + cpu0 node) rather than uname/hostname:
# those two lines (e.g. "milkv,pioneer" board-level, "thead,c920" cpu-level)
# are the one thing actually tied to the silicon and survive a kernel bump or
# a renamed host, unlike `uname -a`'s hostname/kernel-release fields.
PLATFORM_FLAGS :=
ifeq ($(shell grep -qas -e 'sophgo' -e 'thead,c920' -e 'milkv,pioneer' \
        /proc/device-tree/compatible /proc/device-tree/cpus/cpu@0/compatible \
        2>/dev/null && echo yes),yes)
PLATFORM_FLAGS := -DTARGET_MILKV_PIONEER
endif

all: pmu_test pmu_scanner arch_fault_scanner
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f pmu_test pmu_scanner arch_fault_scanner *.csv

# ── kernel module ──────────────────────────────────────────
load:
	sudo insmod pmu_module.ko

unload:
	sudo rmmod pmu_module 2>/dev/null || true

reload: unload load

# ── user-space binaries ────────────────────────────────────
pmu_test: pmu_test.c
	gcc -O2 -o pmu_test pmu_test.c

# Speculative scanner
pmu_scanner: pmu_scanner.c
	gcc -O2 -no-pie $(PLATFORM_FLAGS) -o pmu_scanner pmu_scanner.c

# Architecture fault scanner
arch_fault_scanner: arch_fault_scanner.c
	gcc -O2 -o arch_fault_scanner arch_fault_scanner.c

.PHONY: all clean load unload reload test scan scan_spec verify