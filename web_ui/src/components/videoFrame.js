import React, { Component } from 'react';
import { Grid, FormControlLabel,Switch } from '@material-ui/core';

const STREAM_URL = 'http://127.0.0.1:8000/stream';

class videoFrame extends Component {
  constructor(props) {
    super(props);
    this.state = {
        src: '',
        online: false,
    }
  }

  toggleOn() {
    this.setState({
      online: !this.state.online,
      src: this.state.online ? '' : STREAM_URL,
    });
  }

  render() {
      const frameStyle = {
          height:'360px',
          width:'640px',
      }
    return (<Grid container justify="center" spacing={2}>
        <Grid item xs={12}>
        <FormControlLabel
        control={
          <Switch checked={this.state.online} onChange={()=> this.toggleOn() } />
        }
        label="Video"
      />
        </Grid>
        <Grid item xs={12}>
            <img src={this.state.src} style={frameStyle} alt=''></img>
        </Grid>
    </Grid>)
  }
}

export default videoFrame;
