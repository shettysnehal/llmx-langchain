import { PrismaClient } from '../../../generated/prisma/client';// Adjust the path if needed
import { NextResponse } from 'next/server';
//uploading a new blog or pdf
const prisma = new PrismaClient();

export async function POST(req,res) {
  try {
    const body = await req.json();
    console.log("Received body:", body);
    const { userId, email, topics ,contentId} = body;

    if (!userId || !email || !Array.isArray(topics)) {
      return NextResponse.json({ error: "Missing or invalid data" }, { status: 400 });
    }

    const blogCount = await prisma.blog.count({
      where: { userId },
    });

    const blogTitle = `${email}_${blogCount + 1}`;

    const blog = await prisma.blog.create({
      data: {
        title: blogTitle,
        contentId: contentId,
        user: { connect: { id: userId } },
        topics: {
          create: topics.map(name => ({
            name,
            mcqIpfsUrl: "",
            videoIpfsUrl: "",
          })),
        },
      },
      include: {
        topics: true,
      },
    });

    return NextResponse.json({
      message: "Blog created successfully",
      blogId: blog.id,
      blogTitle: blog.title,
      topics: blog.topics,
    });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
