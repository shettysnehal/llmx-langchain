'use client';
/* lets users upload as many blogs as they want, each blog is a pdf file, and the user can upload a pdf file to the server, which will then be stored in the database. The user can also view all the blogs they have posted.
 */
import { use, useEffect, useState } from 'react';
import { getSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Button } from '../../components/ui/button';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

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
    const contentId = uuidv4(); // generates a unique content ID


   const formData = new FormData();
formData.append('pdf', file);
formData.append('contentId', contentId); 

    try {
      const res = await fetch('http://localhost:5000/api/upload-blog', {
        method: 'POST',
        body: formData,
      });
      
if (!res.ok) {
  const errorText = await res.text(); // safer than .json() if not JSON
  console.error('Upload API failed:', res.status, errorText);
  return;
}

const data = await res.json();
console.log('Upload success:', data);
    
      try{
        if (!session) {
  alert("Session not loaded yet. Please wait.");
  return;
} 
console.log(session)
    const dbRes = await fetch('http://localhost:3000/api/post', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  
  body: JSON.stringify({
    userId: session.userId,
    email: session.user.email,
    topics: data,
    contentId: contentId, // include the contentId in the request
  }),
});
const result = await dbRes.json();
   console.log('Database response:', result);
       const blogId = result.blogId;
       if (blogId) {
        router.push(`/blogs/${blogId}`);}
      else {
          console.error('Blog ID not found in response');
        }
}
catch (error) {
      console.error('Database upload error:', error);}

   
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
      
      {file && (
        <p className="text-sm text-gray-700">Selected file: <span className="font-semibold">{file.name}</span></p>
      )}

      <Button onClick={upload}>Upload your blog</Button>
    </div>
  )
);
}
