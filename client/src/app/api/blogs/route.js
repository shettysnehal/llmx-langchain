import { NextResponse } from 'next/server';
import { PrismaClient } from '@/generated/prisma/client';
//api for displaying all blogs of a user

const prisma = new PrismaClient();


export async function POST(req) {
  const body = await req.json();
  const { id } = body;

  if (!id) {
    return NextResponse.json({ error: "User ID is required" }, { status: 400 });
  }

  try {
    const user = await prisma.user.findUnique({
      where: { id },
      include: {
        blogs: true, // Fetch all blogs related to the user
      },
    });

    if (!user || !user.blogs || user.blogs.length === 0) {
      return NextResponse.json({ error: "No blogs found for this user" }, { status: 404 });
    }

    return NextResponse.json(user.blogs, { status: 200 });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
