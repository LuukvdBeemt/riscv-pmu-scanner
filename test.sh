#!/bin/bash
# test.sh — rebuild, reload, and verify the PMU kernel module
set -e

MODULE=pmu_module
TEST=./pmu_test

# ── 1. Unload old module if present ──────────────────────
echo "[ 1/4 ] Unloading old module..."
sudo rmmod "$MODULE" 2>/dev/null && echo "  Unloaded $MODULE" || echo "  Not loaded, skipping"

# ── 2. Build ──────────────────────────────────────────────
echo ""
echo "[ 2/4 ] Building..."
make clean -s
make -s
echo "  Build OK"

# ── 3. Insert module ──────────────────────────────────────
echo ""
echo "[ 3/4 ] Loading ${MODULE}.ko..."
sudo insmod "${MODULE}.ko"

# Give the module a moment to finish init and write to dmesg
sleep 0.3

echo ""
echo "  --- dmesg (module init) ---"
sudo dmesg | grep -E "PMU" | tail -20
echo "  ---------------------------"

# ── 4. Build and run test ─────────────────────────────────
echo ""
echo "[ 4/4 ] Running pmu_test..."
echo ""
make pmu_test -s
sudo $TEST