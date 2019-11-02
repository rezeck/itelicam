// import react apps
import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid,Paper } from '@material-ui/core';

// import local components
import VideoFrame from './components/videoFrame'
import Navbar from './components/navbar'
import EventsPanel from './components/eventPanel'

const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
  },
  paper: {
    margin: theme.spacing(2),
    padding: theme.spacing(2),
    textAlign: 'center',
    color: theme.palette.text.secondary,
  },
}));

function App() {
  const classes = useStyles();
  return (
    <div className={classes.root}>
      <Navbar></Navbar>
      <Grid container spacing={2} >
        <Grid item xs={6} spacing={2}>
          <Paper className={classes.paper}>
            <VideoFrame></VideoFrame>
          </Paper>
        </Grid>
        <Grid item xs={6}>
          <Paper className={classes.paper}>
            <EventsPanel></EventsPanel>
          </Paper>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
