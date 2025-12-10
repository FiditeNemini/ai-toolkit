import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';

const execAsync = promisify(exec);

export async function GET() {
  try {
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    // Try NVIDIA first (Windows/Linux)
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);
    if (hasNvidiaSmi) {
      const gpuStats = await getGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: true,
        hasAppleSilicon: false,
        gpus: gpuStats,
      });
    }

    // Check for Apple Silicon on macOS
    if (isMac) {
      const appleGpuInfo = await getAppleSiliconGpuInfo();
      if (appleGpuInfo) {
        return NextResponse.json({
          hasNvidiaSmi: false,
          hasAppleSilicon: true,
          gpus: [appleGpuInfo],
        });
      }
    }

    return NextResponse.json({
      hasNvidiaSmi: false,
      hasAppleSilicon: false,
      gpus: [],
      error: 'No supported GPU detected',
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        hasAppleSilicon: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      await execAsync('nvidia-smi -L');
    } else {
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function getGpuStats(isWindows: boolean) {
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

  const gpus = stdout
    .trim()
    .split('\n')
    .map(line => {
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = line.split(', ').map(item => item.trim());

      return {
        index: parseInt(index),
        name,
        driverVersion,
        temperature: parseInt(temperature),
        utilization: {
          gpu: parseInt(gpuUtil),
          memory: parseInt(memoryUtil),
        },
        memory: {
          total: parseInt(memoryTotal),
          free: parseInt(memoryFree),
          used: parseInt(memoryUsed),
        },
        power: {
          draw: parseFloat(powerDraw),
          limit: parseFloat(powerLimit),
        },
        clocks: {
          graphics: parseInt(clockGraphics),
          memory: parseInt(clockMemory),
        },
        fan: {
          speed: parseInt(fanSpeed) || 0,
        },
        isAppleSilicon: false,
      };
    });

  return gpus;
}

interface SystemProfilerGPU {
  _name?: string;
  'sppci_model'?: string;
  'spdisplays_device_type'?: string;
  'spdisplays_vram'?: string;
  'spdisplays_vram_shared'?: string;
}

interface SystemProfilerData {
  SPDisplaysDataType?: SystemProfilerGPU[];
}

async function getAppleSiliconGpuInfo() {
  try {
    // Use system_profiler to get GPU info on macOS
    const { stdout } = await execAsync('system_profiler SPDisplaysDataType -json');
    const data: SystemProfilerData = JSON.parse(stdout);
    const displays = data.SPDisplaysDataType;

    if (!displays || displays.length === 0) return null;

    // Find the Apple GPU (the first one is typically the integrated GPU)
    const appleGpu = displays.find(
      (d: SystemProfilerGPU) => d['sppci_model']?.includes('Apple') || d._name?.includes('Apple'),
    );

    // If no Apple GPU found, just use the first display
    const gpu = appleGpu || displays[0];
    if (!gpu) return null;

    // Get total system memory via sysctl
    let totalSystemMemoryMB = 0;
    try {
      const { stdout: memStdout } = await execAsync('sysctl -n hw.memsize');
      totalSystemMemoryMB = parseInt(memStdout.trim()) / (1024 * 1024);
    } catch {
      // Fallback: default to 16GB if we can't get memory info
      totalSystemMemoryMB = 16384;
    }

    // Apple Silicon uses unified memory - GPU can access most of system RAM
    // Estimate ~75% is available to GPU (rest reserved for system)
    const gpuMemoryMB = Math.floor(totalSystemMemoryMB * 0.75);

    // Extract VRAM info if available (older Macs with dedicated GPUs)
    let reportedVram = 0;
    if (gpu['spdisplays_vram']) {
      const vramMatch = gpu['spdisplays_vram'].match(/(\d+)/);
      if (vramMatch) {
        reportedVram = parseInt(vramMatch[1]);
        // Check if it's in GB
        if (gpu['spdisplays_vram'].toLowerCase().includes('gb')) {
          reportedVram *= 1024;
        }
      }
    }

    const gpuName = gpu['sppci_model'] || gpu._name || 'Apple Silicon GPU';

    return {
      index: 0,
      name: gpuName,
      driverVersion: 'Apple Metal',
      temperature: 0, // Not easily accessible on macOS without sudo
      utilization: {
        gpu: 0, // Would require powermetrics (needs sudo)
        memory: 0,
      },
      memory: {
        total: reportedVram || gpuMemoryMB,
        free: reportedVram || gpuMemoryMB, // Can't easily get real-time usage
        used: 0,
      },
      power: {
        draw: 0,
        limit: 0,
      },
      clocks: {
        graphics: 0,
        memory: 0,
      },
      fan: {
        speed: 0, // Apple Silicon Macs often don't have fans or don't report speed
      },
      isAppleSilicon: true,
    };
  } catch (error) {
    console.error('Error detecting Apple Silicon GPU:', error);
    return null;
  }
}
