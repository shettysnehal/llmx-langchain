import { PrismaClient } from '../../../generated/prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function POST(req) {
  try {
    const body = await req.json();
    const { email } = body;

    if (!email) {
      return new Response(JSON.stringify({ error: 'Email is required' }), {
        status: 400,
      });
    }

    // Check if user exists
    let user = await prisma.user.findUnique({ where: { email } });

    // If not, create new user
    if (!user) {
      user = await prisma.user.create({
        data: { email },
      });
    }

    // Generate JWT token with email & id
    const token = jwt.sign(
      { email: user.email, id: user.id },
      process.env.JWT_SECRET,
      { expiresIn: '4d' }
    );

    return new Response(JSON.stringify({ accessToken: token, userId: user.id }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Login error:', error);
    return new Response(JSON.stringify({ error: 'Internal Server Error' }), {
      status: 500,
    });
  }
}
