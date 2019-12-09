# Measure CPU Time Stamp on A Ubuntu Server

##### System Configuration
1. Edit `/etc/default/grub` as:
  ```
  GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_pstate=disable isolcpus=7 nohz_full=7 nosmep nosmap"
  ```
  to disable Intel PState and to prevent system from using core 7.
  Then `sudo update-grub && sudo reboot`.
1. Change BIOS settings:
  - Enable `OS DBMP` or OS control, and enable C_State.
  - Optionally disable hyper-threading on all cores.
1. Set CPU Frequency:
  - Check if module `acpi_cpuinfo` is loaded: `lsmod | grep acpi`.
  - Check CPUFreq: `lscpu | grep MHz`.
  - An easy way using `cpupower` (`sudo apt install linux-tools-common`):
    - `cpupower frequency-info` shows available frequency settings.
    - `cpupower frequency-set` sets CPUs to run in the `<freq>` in **KHz**:
      - append `--min <freq>` or `--max <freq>` to set a range;
      - append `--freq <freq>` to set a specific frequency.
      - Example: `cpupower frequency-set --freq 2600000`.
  - A complex but more flexible way:
    ```
    cd /sys/devices/system/cpu/
    echo performance | tee cpu*/cpufreq/scaling_governor
    // check available frequencies
    cat cpu0/cpufreq/scaling_available_frequencies
    // <freq> is in KHz, from availabel frequencies
    echo <freq> | tee cpu*/cpufreq/scaling_max_freq
    echo <freq> | tee cpu*/cpufreq/scaling_min_freq
    // check if it works
    lscpu | grep MHz
    ```

1. Disable Hyper-threading on core 7: `echo 0 > /sys/devices/system/cpu/cpu15/online`.
1. Set IRQ affinity mask to CPU 0: `echo 1 | tee /proc/irq/*/smp_affinity`.
1. Run command on CPU 7: `sudo taskset -c 7 <command>`

##### Code to measure CPU Time Stamp

This code takes 64-bit measurements.

```c++
#include <time.h>

// either
#include <stdint.h>
// or use unsigned int / unsigned long int

// saves 64-bit CPU time stamp into `time`
inline void GetCPUTime(uint64_t& time) {
  unsigned int *val = (unsigned int *)time;
  asm volatile ("rdtscp" : "=a" (val[0]), "=d" (val[1])); // no mfence, lfence
}

uint64_t time_start, time_stop, time_elapsed;
GetCPUTime(time_start);
// ... target code ...
GetCPUTime(time_stop);
time_elapsed = time_stop - time_start; // roughly 20 if code is empty
```
