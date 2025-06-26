import { NextResponse } from 'next/server';
import { PrismaClient } from '@/generated/prisma/client';
// API for displaying a single blog by its ID
//also displays the topics of the blog and the user who created the blog

const prisma = new PrismaClient();

export async function GET(req,{params}) {
  
  const {blogId } = params
  console.log(blogId)
    
  

  if (!blogId) {
    return NextResponse.json({ error: "Blog ID is required" }, { status: 400 });
  }

  try {
    const blog = await prisma.blog.findUnique({
      where: { id: blogId },
      include: {
        topics: true,
        user: true,
      },
    });
    console.log(blog)

    if (!blog) {
      return NextResponse.json({ error: "Blog not found" }, { status: 404 });
    }

    return NextResponse.json(blog, { status: 200 });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
