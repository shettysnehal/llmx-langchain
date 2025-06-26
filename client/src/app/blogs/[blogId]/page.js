'use client';
/* displays the each topic of the blog and allows user to generate video for each topic */

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../../components/ui/table";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { use } from "react";


export default function BlogPage({ params }) {
  const router = useRouter();
   const { blogId } = use(params); 
  const [blog, setBlog] = useState(null);
  const [loadingTopicId, setLoadingTopicId] = useState(null);

  useEffect(() => {
    const fetchBlog = async () => {
      const res = await fetch(`http://localhost:3000/api/blogs/${blogId}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
      });
      
      const data = await res.json();
      if(!data){
        router.push('/blogs');
        return;
      }
      setBlog(data);
    };

    fetchBlog();
  }, [blogId]);

  const handleGenerateVideo = async (topic) => {
    try {
      setLoadingTopicId(topic.id);
      console.log(topic.name, blog.contentId);
      // Step 1: Generate IPFS for video
      const genRes = await fetch("http://localhost:5000/api/generate-vid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: topic.name,contentId: blog.contentId }),
      });
      console.log("genRes", genRes);

      const { ipfs } = await genRes.json();
      if (!ipfs) throw new Error("No IPFS returned");

      // Attach video IPFS to topic in db 
      const attachRes = await fetch(`https://files.lighthouse.storage/viewFile/${ipfs}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topicId: topic.id }),
      });

      if (!attachRes.ok) throw new Error("Failed to attach video IPFS");

      if (attachRes.status === 200) {
        router.push(`/blogs/${blogId}/${ipfs}/vid`);
      }

        
       
    } catch (err) {
      console.error(err);
      alert("Error generating video");
    } finally {
      setLoadingTopicId(null);
    }
  };

  if (!blog) return <p>Loading...</p>;

  return (
    <Table>
      <TableCaption>Topics extracted from {blog.title}</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Topic Name</TableHead>
          
          <TableHead>Action</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {blog.topics.map((topic) => (
          <TableRow key={topic.id}>
            <TableCell className="font-medium">{topic.name}</TableCell>
            
            <TableCell>
              {topic.videoIpfsUrl ?(<Button
                onClick={() => router.push(`https://files.lighthouse.storage/viewFile/${topic.videoIpfsUrl}`)}> View Video</Button>
              ):(
                <Button
                  onClick={() => handleGenerateVideo(topic)}
                  disabled={loadingTopicId === topic.id}
                >
                  {loadingTopicId === topic.id ? "Generating..." : "Generate Video"}
                </Button>

              )
              }
             
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
