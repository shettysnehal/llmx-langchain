'use client';

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { use } from 'react';

export default function LighthouseVideoViewer({ params }) {
  const { ipfs } = use(params);
  const fileUrl = `https://files.lighthouse.storage/viewFile/${ipfs}`;
  const [videoSrc, setVideoSrc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloadUrl, setDownloadUrl] = useState(null);

  useEffect(() => {
    const fetchVideo = async () => {
      try {
        const res = await fetch(fileUrl);
        const contentType = res.headers.get('Content-Type');
        console.log('Content-Type:', contentType); // debug MIME type

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setVideoSrc(url);
        setDownloadUrl(url);
      } catch (err) {
        console.error('Failed to load video:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchVideo();
  }, []);

  return (
    <Card className="w-full max-w-3xl mx-auto p-4">
      <CardHeader className="text-xl font-semibold">
        Lighthouse Video Preview
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="w-full h-[400px] rounded-lg" />
        ) : videoSrc ? (
          <video
            src={videoSrc}
            controls
            type="video/mp4"
            className="w-full rounded-lg shadow-md"
          />
        ) : (
          <p className="text-red-500">Unable to load video</p>
        )}

        {downloadUrl && (
          <Button className="mt-4">
            <a href={downloadUrl} download="video.mp4">
              Download Video
            </a>
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
