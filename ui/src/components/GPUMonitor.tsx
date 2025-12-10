import React, { useState, useEffect, useRef, useMemo } from 'react';
import { GPUApiResponse } from '@/types';
import Loading from '@/components/Loading';
import GPUWidget from '@/components/GPUWidget';
import { apiClient } from '@/utils/api';

const GpuMonitor: React.FC = () => {
  const [gpuData, setGpuData] = useState<GPUApiResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const isFetchingGpuRef = useRef(false);

  useEffect(() => {
    let isMounted = true;
    let abortController: AbortController | null = null;

    const fetchGpuInfo = async () => {
      if (isFetchingGpuRef.current || !isMounted) {
        return;
      }
      setLoading(true);
      isFetchingGpuRef.current = true;

      // Create new abort controller for this request
      abortController = new AbortController();

      try {
        const res = await apiClient.get('/api/gpu', { signal: abortController.signal });
        if (isMounted) {
          setGpuData(res.data);
          setLastUpdated(new Date());
          setError(null);
        }
      } catch (err: unknown) {
        // Ignore abort/cancel errors completely
        if (
          !isMounted ||
          (err instanceof Error && (
            err.name === 'AbortError' ||
            err.name === 'CanceledError' ||
            err.message === 'canceled' ||
            err.message.includes('cancel')
          )) ||
          (typeof err === 'object' && err !== null && 'code' in err && err.code === 'ERR_CANCELED')
        ) {
          return;
        }
        if (isMounted) {
          setError(`Failed to fetch GPU data: ${err instanceof Error ? err.message : String(err)}`);
        }
      } finally {
        isFetchingGpuRef.current = false;
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    // Fetch immediately on component mount
    fetchGpuInfo();

    // Set up interval to fetch every 1 seconds
    const intervalId = setInterval(fetchGpuInfo, 1000);

    // Clean up interval and abort any in-flight requests on component unmount
    return () => {
      isMounted = false;
      clearInterval(intervalId);
      if (abortController) {
        abortController.abort();
      }
    };
  }, []);

  const getGridClasses = (gpuCount: number): string => {
    switch (gpuCount) {
      case 1:
        return 'grid-cols-1';
      case 2:
        return 'grid-cols-2';
      case 3:
        return 'grid-cols-3';
      case 4:
        return 'grid-cols-4';
      case 5:
      case 6:
        return 'grid-cols-3';
      case 7:
      case 8:
        return 'grid-cols-4';
      case 9:
      case 10:
        return 'grid-cols-5';
      default:
        return 'grid-cols-3';
    }
  };

  console.log('state', {
    loading,
    gpuData,
    error,
    lastUpdated,
  });

  const content = useMemo(() => {
    if (loading && !gpuData) {
      return <Loading />;
    }

    if (error) {
      return (
        <div className="bg-red-900 border border-red-600 text-red-200 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      );
    }

    if (!gpuData) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">No GPU data available.</span>
        </div>
      );
    }

    if (!gpuData.hasNvidiaSmi && !gpuData.hasAppleSilicon) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">No supported GPUs detected!</strong>
          <span className="block sm:inline"> No NVIDIA or Apple Silicon GPU found.</span>
          {gpuData.error && <p className="mt-2 text-sm">{gpuData.error}</p>}
        </div>
      );
    }

    if (gpuData.gpus.length === 0) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">No GPUs found, but nvidia-smi is available.</span>
        </div>
      );
    }

    const gridClass = getGridClasses(gpuData?.gpus?.length || 1);

    return (
      <div className={`grid ${gridClass} gap-3`}>
        {gpuData.gpus.map((gpu, idx) => (
          <GPUWidget key={idx} gpu={gpu} />
        ))}
      </div>
    );
  }, [loading, gpuData, error]);

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <h1 className="text-md">GPU Monitor</h1>
        <div className="text-xs text-gray-500">Last updated: {lastUpdated?.toLocaleTimeString()}</div>
      </div>
      {content}
    </div>
  );
};

export default GpuMonitor;
