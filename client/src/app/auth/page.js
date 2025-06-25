'use client';

import { signIn } from 'next-auth/react';
import { Button } from '../../components/ui/button';

export default function ButtonDemo() {
  return (
    
    <div className="flex items-center justify-center min-h-screen p-8">
      <Button onClick={() => signIn('google')}>
        Continue with Google
      </Button>
    </div>
  );
}
