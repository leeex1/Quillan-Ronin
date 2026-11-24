import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const MMORPGWorld: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">MMORPG World - SocialFabric</Typography>
        <Typography>
          JSSC 500-player simulation with Rust backend
        </Typography>
      </CardContent>
    </Card>
  );
};

export default MMORPGWorld;