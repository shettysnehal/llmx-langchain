import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import {jwtDecode} from "jwt-decode";
import axios from "axios";
//goal:match the expiration of cookie with jwt

const handler = NextAuth({
  session: {
    strategy: "jwt",
    maxAge: 7 * 24 * 60 * 60,
  },

  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
  ],

  secret: process.env.NEXTAUTH_SECRET,

  callbacks: {
    async jwt({ token, account, profile }) {
      if (account && profile?.email) {
        const { data } = await axios.post(`${process.env.BACKEND_URL}/api/login`, {
          email: profile.email,
        });
        console.log("Token received:", data.accessToken);
        const decoded = jwtDecode(data.accessToken);

        return {
          ...token,
          accessToken: data.accessToken,
          email: profile.email,
          userId: data.userId,
          iat: decoded.iat,
          exp: decoded.exp,
        };
      }
      return token;
    },

    async session({ session, token }) {
      session.accessToken = token.accessToken;
      session.email = token.email;
      session.userId = token.userId;
      return session;
    },

    redirect() {
      return '/upload';
    },
  },
});

// ✅ Required in App Router!
export { handler as GET, handler as POST };
