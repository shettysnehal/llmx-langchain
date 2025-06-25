'use client';

import { useEffect, useState } from 'react';
import { getSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Button } from '../../components/ui/button';

export default function Page() {
  const [session, setSession] = useState(null);
  const [file, setFile] = useState(null);
  const router = useRouter();

  useEffect(() => {
    getSession().then((data) => {
      if (!data) {
        router.push('/auth');
      } else {
        setSession(data);
      }
    });
  }, [router]);

  const upload = async () => {
    if (!file) {
      alert('Please select a PDF file first.');
      return;
    }

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      const res = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      console.log('Upload success:', data);
    } catch (err) {
      console.error('Upload error:', err);
    }
  };

  return (
    session && (
       <div className="flex flex-col items-center justify-center min-h-screen p-8 gap-4">
        <label className="text-black font-medium">Choose a PDF file to upload:</label>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block text-black mb-4 p-2 border border-gray-300 rounded"
        />
        <Button onClick={upload}>Upload your blog</Button>
      </div>
    )
  );
}
