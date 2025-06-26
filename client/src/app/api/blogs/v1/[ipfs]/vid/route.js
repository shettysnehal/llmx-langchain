import { NextResponse } from 'next/server';
import { PrismaClient } from '@/generated/prisma/client';

const prisma = new PrismaClient();

export async function POST(req,{params}) {
  console.log(params)
 
  const {ipfs}= params;

  if (!ipfs) {
    return NextResponse.json({ error: "IPFS hash is required" }, { status: 400 });
  }

  const body = await req.json();
  const { topicId } = body;

  if (!topicId) {
    return NextResponse.json({ error: "Topic ID is required" }, { status: 400 });
  } 

  try {
    const updatedTopic = await prisma.topic.create({
      where: { id: topicId },
      data: {
        videoIpfsUrl: ipfs,
      },
    });

    return NextResponse.json(updatedTopic, { status: 200 });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Failed to update topic" }, { status: 500 });
  }
}
